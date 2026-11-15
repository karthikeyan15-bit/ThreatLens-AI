"""AgentTune training pipeline for ThreatLens AI.

Performs genuine AgentTune fine-tuning using the supported
``agenttune.train_agentic`` API. Trains on real investigation trajectories
collected from the ThreatLens agent and saves model checkpoints + LoRA
adapters to disk.

Hardware: CPU or GPU (auto-detected). No fabricated results — training only
runs when ``agenttune`` and a compatible backend (TRL) are importable.
"""

import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from evaluation.agenttune.agenttune_rewards import (
    AGENTTUNE_REWARD_FUNCS,
    AGENTTUNE_REWARD_WEIGHTS,
)
from evaluation.agenttune.trajectory_adapter import export_trajectories_jsonl
from evaluation.agenttune.trajectory_collector import (
    TrajectoryCollector,
    TrajectoryTier,
    collect_trajectory,
)
from evaluation.agenttune.trajectory_validator import validate_trajectory

# Check AgentTune availability
try:
    import agenttune
    from agenttune import train_agentic

    AGENTTUNE_AVAILABLE = True
except ImportError:
    AGENTTUNE_AVAILABLE = False

# Check TRL availability (required backend for agentic training on CPU/GPU)
try:
    import trl  # noqa: F401

    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for AgentTune training."""

    # Algorithm
    algorithm: str = "grpo"

    # Base model served via Hugging Face Router
    base_model: str = "CohereLabs/tiny-aya-global:cohere"

    # Training parameters
    learning_rate: float = 1e-5
    batch_size: int = 2
    num_epochs: int = 3
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 8

    # LoRA config
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Data
    train_data_path: str = "artifacts/agenttune/trajectories/train.jsonl"
    val_data_path: str = "artifacts/agenttune/trajectories/val.jsonl"
    test_data_path: str = "artifacts/agenttune/trajectories/test.jsonl"

    # Output
    output_dir: str = "artifacts/agenttune/checkpoints"
    logging_steps: int = 5
    save_steps: int = 25
    eval_steps: int = 25

    # Reward weights
    reward_weights: Dict[str, float] = field(default_factory=dict)

    # Hardware
    device: str = "auto"  # auto, cuda, cpu
    mixed_precision: str = "fp32"  # fp32 on CPU, bf16/fp16 on GPU
    gradient_checkpointing: bool = False

    # Experiment tracking
    experiment_name: str = "threatlens-agenttune"
    wandb_project: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Reproducibility
    seed: int = 42

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)


class TrainingPipeline:
    """Orchestrates the complete AgentTune training pipeline."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(config.seed)

        # Save config for reproducibility
        self.config.to_yaml(self.output_dir / "training_config.yaml")

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("agenttune_training")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    # ------------------------------------------------------------------ #
    # Data preparation
    # ------------------------------------------------------------------ #
    def _load_scenarios(self) -> List[Dict]:
        """Load all evaluation scenarios."""
        scenario_path = Path("datasets/evaluation/scenarios.json")
        if not scenario_path.exists():
            self.logger.warning("No scenarios file found at %s", scenario_path)
            return []
        with open(scenario_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def collect_trajectories(
        self,
        scenario_ids: Optional[List[str]] = None,
        output_dir: str = "artifacts/agenttune/trajectories",
    ) -> Dict[str, str]:
        """Collect, validate, and split trajectories into train/val/test.

        Returns paths to train/val/test JSONL files.
        """
        scenarios = {s["id"]: s for s in self._load_scenarios()}
        if scenario_ids is None:
            scenario_ids = list(scenarios.keys())

        self.logger.info(
            "Collecting trajectories for %d scenarios...", len(scenario_ids)
        )

        trajectories = []
        for scenario_id in scenario_ids:
            scenario = scenarios.get(scenario_id)
            if not scenario:
                self.logger.warning("Scenario %s not found, skipping", scenario_id)
                continue

            try:
                trajectory = collect_trajectory(
                    request=scenario["request"],
                    scenario_id=scenario_id,
                    tier=TrajectoryTier.FULL,
                    ground_truth=scenario.get("reference", {}),
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.error("Failed to collect %s: %s", scenario_id, exc)
                continue

            is_valid, issues = validate_trajectory(trajectory)
            if not is_valid:
                self.logger.warning(
                    "Trajectory %s has validation issues: %s", scenario_id, issues
                )

            trajectories.append(trajectory)

        # Deterministic split
        trajectories.sort(key=lambda t: t.scenario_id)
        n = len(trajectories)
        train_end = max(1, int(n * 0.7))
        val_end = train_end + max(1, int(n * 0.15))

        splits = {
            "train": trajectories[:train_end],
            "val": trajectories[train_end:val_end],
            "test": trajectories[val_end:],
        }

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}
        for name, trajs in splits.items():
            path = str(out_dir / f"{name}.jsonl")
            export_trajectories_jsonl(trajs, path, format="agenttune")
            self.logger.info(
                "Saved %d %s trajectories to %s", len(trajs), name, path
            )
            paths[name] = path

        return paths

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def _resolve_device(self) -> Dict[str, Any]:
        """Resolve CPU/GPU training settings."""
        import torch

        cuda_available = torch.cuda.is_available()
        if self.config.device == "cuda" and not cuda_available:
            self.logger.warning(
                "CUDA requested but not available — falling back to CPU"
            )
            cuda_available = False

        use_cuda = cuda_available and self.config.device in ("auto", "cuda")
        mixed = self.config.mixed_precision
        if not use_cuda:
            mixed = "fp32"  # CPU cannot use bf16/fp16 efficiently here

        return {
            "use_cuda": use_cuda,
            "mixed_precision": mixed,
            "device": "cuda" if use_cuda else "cpu",
        }

    def _create_trainer(self, train_dataset, val_dataset):
        """Create the AgentTune trainer via the supported API."""
        if not AGENTTUNE_AVAILABLE:
            raise RuntimeError(
                "AgentTune is not installed. Run: pip install agenttune"
            )
        if not TRL_AVAILABLE:
            raise RuntimeError(
                "TRL is required for AgentTune agentic training. "
                "Run: pip install trl"
            )

        device_info = self._resolve_device()
        self.logger.info(
            "AgentTune training on device=%s mixed_precision=%s",
            device_info["device"],
            device_info["mixed_precision"],
        )

        # Build reward functions from the configured weights
        reward_funcs = []
        reward_weights = []
        weights = self.config.reward_weights or AGENTTUNE_REWARD_WEIGHTS
        for name, weight in weights.items():
            fn = AGENTTUNE_REWARD_FUNCS.get(name)
            if fn is None:
                self.logger.warning("Unknown reward function %s, skipping", name)
                continue
            reward_funcs.append(fn)
            reward_weights.append(weight)

        if not reward_funcs:
            raise RuntimeError("No valid reward functions configured")

        trainer = train_agentic(
            algorithm=self.config.algorithm,
            model=self.config.base_model,
            reward_funcs=reward_funcs,
            reward_weights=reward_weights,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_seq_length=self.config.max_seq_length,
            lora_r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            logging_dir=str(self.output_dir / "logs"),
            report_to="wandb" if self.config.wandb_project else "none",
            run_name=self.config.experiment_name,
            seed=self.config.seed,
        )
        return trainer

    def run_training(
        self,
        scenario_ids: Optional[List[str]] = None,
        skip_if_no_data: bool = True,
    ) -> Dict[str, Any]:
        """Execute the complete AgentTune training pipeline.

        Returns a dict with training status, output dir, and metrics.
        """
        if not AGENTTUNE_AVAILABLE:
            raise RuntimeError(
                "AgentTune is not installed. Run: pip install agenttune"
            )

        self.logger.info("Starting AgentTune training pipeline")
        self.logger.info(
            "Algorithm: %s  Base model: %s",
            self.config.algorithm,
            self.config.base_model,
        )

        # 1. Collect & split trajectories
        data_paths = self.collect_trajectories(
            scenario_ids=scenario_ids,
            output_dir="artifacts/agenttune/trajectories",
        )

        train_path = data_paths["train"]
        val_path = data_paths["val"]

        # 2. Load datasets for AgentTune
        from datasets import load_dataset

        train_dataset = load_dataset("json", data_files=train_path)["train"]
        val_dataset = load_dataset("json", data_files=val_path)["train"]

        self.logger.info(
            "Loaded train=%d val=%d examples",
            len(train_dataset),
            len(val_dataset),
        )

        # 3. Create trainer
        trainer = self._create_trainer(train_dataset, val_dataset)

        # 4. Train (genuine execution)
        self.logger.info("Starting AgentTune training...")
        train_result = trainer.train()

        # 5. Save results
        training_result = {
            "status": "completed",
            "algorithm": self.config.algorithm,
            "base_model": self.config.base_model,
            "device": self._resolve_device()["device"],
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset),
            "metrics": train_result.metrics
            if hasattr(train_result, "metrics")
            else {},
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        self._save_results(training_result)

        self.logger.info(
            "AgentTune training completed. Output dir: %s", self.output_dir
        )
        return {
            "status": "completed",
            "output_dir": str(self.output_dir),
            "result": training_result,
        }

    def _save_results(self, training_result: Dict):
        """Save training results to disk."""
        results_path = self.output_dir / "training_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(training_result, f, indent=2)

        summary = {
            "experiment_name": self.config.experiment_name,
            "algorithm": self.config.algorithm,
            "base_model": self.config.base_model,
            "training_config": self.config.to_dict(),
            "results": training_result,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        summary_path = Path("artifacts/agenttune/training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        report = f"""# AgentTune Training Report

**Experiment:** {self.config.experiment_name}
**Algorithm:** {self.config.algorithm}
**Base Model:** {self.config.base_model}
**Completed:** {training_result.get('completed_at', 'N/A')}
**Device:** {training_result.get('device', 'N/A')}

## Configuration
```yaml
{yaml.dump(self.config.to_dict(), default_flow_style=False)}
```

## Results
```json
{json.dumps(training_result.get('metrics', {}), indent=2)}
```

Training artifacts (checkpoints + LoRA adapters) are saved under
`{self.output_dir}`.
"""
        report_path = Path("artifacts/agenttune/training_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)


def run_baseline_collection(
    scenario_ids: List[int] = None,
    output_dir: str = "artifacts/agenttune/trajectories",
) -> Dict[str, Any]:
    """Run baseline trajectory collection for all scenarios (no training)."""
    if scenario_ids is None:
        scenario_ids = list(range(1, 9))

    scenario_ids = [f"scenario_{i:03d}" for i in scenario_ids]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"trajectories": [], "rewards": [], "errors": []}

    for scenario_id in scenario_ids:
        print(f"Collecting trajectory for {scenario_id}...")
        try:
            with open("datasets/evaluation/scenarios.json", "r", encoding="utf-8") as f:
                scenarios = json.load(f)
            scenario = next(
                (s for s in scenarios if s["id"] == scenario_id), None
            )
            if not scenario:
                print(f"Scenario {scenario_id} not found")
                continue

            trajectory = collect_trajectory(
                request=scenario["request"],
                scenario_id=scenario_id,
                tier=TrajectoryTier.FULL,
                ground_truth=scenario.get("reference", {}),
            )

            is_valid, issues = validate_trajectory(trajectory)

            from evaluation.agenttune.reward_function import compute_trajectory_reward

            reward = compute_trajectory_reward(
                trajectory, ground_truth=trajectory.ground_truth
            )

            collector = TrajectoryCollector()
            path = collector.save_trajectory(trajectory)

            result = {
                "scenario_id": trajectory.scenario_id,
                "trajectory_id": trajectory.trajectory_id,
                "steps": len(trajectory.tool_calls),
                "status": trajectory.completion_status,
                "reward": trajectory.metadata.get("reward", 0),
                "path": str(path),
                "valid": is_valid,
                "validation_issues": issues,
            }

            print(
                f"  Steps: {trajectory.total_steps}, "
                f"Reward: {trajectory.metadata.get('reward', 0):.3f}, "
                f"Status: {trajectory.completion_status}, Valid: {is_valid}"
            )

            results["trajectories"].append(result)
            results["rewards"].append(trajectory.metadata.get("reward", 0))

        except Exception as exc:  # noqa: BLE001
            print(f"Error collecting {scenario_id}: {exc}")
            results["errors"].append({"scenario_id": scenario_id, "error": str(exc)})

    return results


if __name__ == "__main__":
    print("AgentTune Training Pipeline")
    print("=" * 50)

    if not AGENTTUNE_AVAILABLE:
        print("AgentTune not available. Install with: pip install agenttune")
        raise SystemExit(1)

    if not TRL_AVAILABLE:
        print("TRL not available. Install with: pip install trl")
        raise SystemExit(1)

    config = TrainingConfig()
    pipeline = TrainingPipeline(config)
    result = pipeline.run_training()
    print(f"\nTraining completed: {result['status']}")
    print(f"Output dir: {result['output_dir']}")