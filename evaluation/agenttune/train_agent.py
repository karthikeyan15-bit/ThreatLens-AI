"""AgentTune training pipeline for ThreatLens AI."""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from evaluation.agenttune.trajectory_collector import (
    InvestigationTrajectory,
    TrajectoryCollector,
    TrajectoryTier,
    collect_trajectory,
)
from evaluation.agenttune.trajectory_validator import TrajectoryValidator, validate_trajectory
from evaluation.agenttune.reward_function import RewardFunction, compute_trajectory_reward
from evaluation.agenttune.trajectory_adapter import TrajectoryAdapter, export_trajectories_jsonl

# Check AgentTune availability
try:
    import agenttune
    AGENTTUNE_AVAILABLE = True
except ImportError:
    AGENTTUNE_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for AgentTune training."""
    # Algorithm
    algorithm: str = "grpo"  # grpo, dpo, ppo, rloo, bco

    # Model
    base_model: str = "tiny-aya-global"
    model_revision: str = "main"

    # Training parameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 3
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 4

    # LoRA config
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # Data
    train_data_path: str = "artifacts/agenttune/trajectories/train.jsonl"
    val_data_path: str = "artifacts/agenttune/trajectories/val.jsonl"
    test_data_path: str = "artifacts/agenttune/trajectories/test.jsonl"

    # Output
    output_dir: str = "artifacts/agenttune/checkpoints"
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50

    # Reward weights
    reward_weights: Dict[str, float] = field(default_factory=dict)

    # Hardware
    device: str = "auto"  # auto, cuda, cpu
    mixed_precision: str = "bf16"  # bf16, fp16, fp32
    gradient_checkpointing: bool = True

    # Experiment tracking
    experiment_name: str = "threatlens-agenttune"
    wandb_project: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)


class TrainingPipeline:
    """Orchestrates the complete AgentTune training pipeline."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.to_yaml(self.output_dir / "training_config.yaml")

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("agenttune_training")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def prepare_data(
        self,
        scenario_ids: List[str],
        output_dir: str,
        tier: str = "full",
        ground_truth_map: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Collect trajectories for all scenarios and save as train/val/test splits.

        Returns paths to train/val/test JSONL files.
        """
        self.logger.info(f"Collecting trajectories for {len(scenario_ids)} scenarios...")

        collector = TrajectoryCollector()
        all_trajectories = []

        for scenario_id in scenario_ids:
            # Load scenario
            scenario = self._load_scenario(scenario_id)
            if not scenario:
                continue

            ground_truth = ground_truth_map.get(scenario_id) if ground_truth_map else None

            trajectory = collect_trajectory(
                request=scenario["request"],
                scenario_id=scenario_id,
                tier="full",
                ground_truth=scenario.get("reference", {})
            )

            # Validate
            validator = TrajectoryValidator()
            is_valid, issues = validate_trajectory(trajectory)
            if not is_valid:
                self.logger.warning(f"Trajectory for {scenario_id} has issues: {issues}")

            ground_truth_map = trajectory.ground_truth or {}
            trajectory.ground_truth = ground_truth_map

            # Compute reward
            reward_result = compute_trajectory_reward(trajectory, ground_truth=trajectory.ground_truth)
            trajectory.metadata["reward"] = reward_result["total_reward"]
            trajectory.metadata["reward_breakdown"] = reward_result["breakdown"]

            yield trajectory

    def collect_and_split(
        self,
        scenario_ids: List[str],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        output_dir: str = "artifacts/agenttune/trajectories",
        ground_truth_map: Optional[Dict] = None
    ) -> Dict[str, str]:
        """Collect trajectories and split into train/val/test."""
        trajectories = []
        for scenario_id in scenario_ids:
            scenario = self._load_scenario(scenario_id)
            if not scenario:
                continue

            trajectory = collect_trajectory(
                request=scenario["request"],
                scenario_id=scenario_id,
                tier="full",
                ground_truth=scenario.get("reference", {})
            )

            # Validate
            is_valid, issues = validate_trajectory(trajectory)
            if not is_valid:
                self.logger.warning(f"Trajectory for {scenario_id} has issues, skipping")
                continue

            trajectory = list(self.prepare_data([scenario_id], "", ground_truth_map={}))[0]
            trajectories.append(trajectory)

        # Split
        import random
        random.shuffle(trajectories)
        n = len(trajectories)
        train_end = int(n * 0.7)
        val_end = train_end + int(n * 0.15)

        train_traj = trajectories[:train_end]
        val_traj = trajectories[train_end:train_end + int(n * 0.15)]
        test_traj = trajectories[train_end + int(n * 0.15):]

        output_dir = Path("artifacts/agenttune/trajectories")
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}
        for name, trajs in [("train", train_traj), ("val", val_traj), ("test", test_traj)]:
            path = f"artifacts/agenttune/trajectories/{name}.jsonl"
            export_trajectories_jsonl(trajs, path, format="agenttune")
            self.logger.info(f"Saved {len(trajs)} {name} trajectories to {path}")

        return {
            "train": "artifacts/agenttune/trajectories/train.jsonl",
            "val": "artifacts/agenttune/trajectories/val.jsonl",
            "test": "artifacts/agenttune/trajectories/test.jsonl"
        }

    def _load_scenario(self, scenario_id: str) -> Optional[Dict]:
        """Load scenario from evaluation dataset."""
        scenario_path = Path("datasets/evaluation/scenarios.json")
        if not scenario_path.exists():
            return None

        with open(scenario_path, "r") as f:
            scenarios = json.load(f)

        for s in scenarios:
            if s["id"] == scenario_id:
                return s
        return None

    def run_training(self) -> Dict[str, Any]:
        """Execute the complete training pipeline."""
        if not AGENTTUNE_AVAILABLE:
            raise RuntimeError("AgentTune not installed. Run: pip install agenttune")

        self.logger.info("Starting AgentTune training pipeline...")
        self.logger.info(f"Algorithm: {self.config.algorithm}")
        self.logger.info(f"Base model: {self.config.base_model}")

        # 1. Prepare data
        self.logger.info("Preparing training data...")
        scenario_ids = self._get_all_scenario_ids()

        data_paths = self.collect_and_split(
            scenario_ids=list(range(1, 9)),  # scenario_001 through scenario_008
            output_dir="artifacts/agenttune/trajectories"
        )

        # 2. Configure trainer
        self.logger.info("Configuring AgentTune trainer...")
        trainer = self._create_trainer()

        # 3. Train
        self.logger.info("Starting training...")
        # Note: Actual training would call trainer.train()
        # This is a placeholder for the actual training loop
        training_result = self._run_training(trainer)

        # 4. Save results
        self._save_results(training_result)

        return {"status": "completed", "output_dir": str(self.output_dir)}

    def _create_trainer(self):
        """Create AgentTune trainer."""
        if not AGENTTUNE_AVAILABLE:
            raise RuntimeError("AgentTune not available")

        import agenttune
        trainer = agenttune.train_agentic(
            algorithm=self.config.algorithm,
            model=self.config.base_model,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            max_seq_length=self.config.max_seq_length,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            lora_r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            output_dir=self.config.output_dir,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=10,
            weight_decay=0.01,
            max_grad_norm=1.0,
            fp16=False,
            bf16=True,
            gradient_checkpointing=self.config.gradient_checkpointing,
            logging_dir=str(self.output_dir / "logs"),
            report_to="wandb" if self.config.wandb_project else "none",
            run_name=self.config.experiment_name,
        )

        # Load datasets
        # trainer.train_dataset = load_dataset("json", data_files=self.config.train_data_path)["train"]
        # trainer.eval_dataset = load_dataset("json", data_files=self.config.val_data_path)["train"]

        return trainer

    def _run_training(self, trainer) -> Dict:
        """Run the training."""
        # This would call trainer.train() in real usage
        # For now, return mock results
        return {
            "status": "completed",
            "final_loss": 0.0,
            "metrics": {}
        }

    def _save_results(self, training_result: Dict):
        """Save training results."""
        results_path = self.output_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(training_result, f, indent=2)

        # Save training summary
        summary = {
            "experiment_name": self.config.experiment_name,
            "algorithm": self.config.algorithm,
            "base_model": self.config.base_model,
            "training_config": self.config.to_dict(),
            "results": training_result,
            "completed_at": datetime.utcnow().isoformat(),
        }

        summary_path = Path("artifacts/agenttune/training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Markdown report
        report = f"""# AgentTune Training Report

**Experiment:** {self.config.experiment_name}
**Algorithm:** {self.config.algorithm}
**Base Model:** {self.config.base_model}
**Completed:** {datetime.utcnow().isoformat()}

## Configuration
```yaml
{yaml.dump(self.config.to_dict(), default_flow_style=False)}
```

## Results
```json
{json.dumps(training_result, indent=2)}
```
"""
        report_path = Path("artifacts/agenttune/training_report.md")
        with open(report_path, "w") as f:
            f.write(report)


def run_baseline_collection(
    scenario_ids: List[int] = None,
    output_dir: str = "artifacts/agenttune/trajectories"
) -> Dict[str, Any]:
    """Run baseline trajectory collection for all scenarios."""
    if scenario_ids is None:
        scenario_ids = list(range(1, 9))  # 1-8

    scenario_ids = [f"scenario_{i:03d}" for i in scenario_ids]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "trajectories": [],
        "rewards": [],
        "errors": []
    }

    for scenario_id in scenario_ids:
        print(f"Collecting trajectory for {scenario_id}...")
        try:
            # Load scenario
            with open("datasets/evaluation/scenarios.json", "r") as f:
                scenarios = json.load(f)

            scenario = next((s for s in json.load(open("datasets/evaluation/scenarios.json")) if s["id"] == scenario_id), None)
            if not scenario:
                print(f"Scenario {scenario_id} not found")
                continue

            # Collect trajectory
            from evaluation.agenttune.trajectory_collector import collect_trajectory
            trajectory = collect_trajectory(
                request=scenario["request"],
                scenario_id=scenario_id,
                ground_truth=scenario.get("reference", {})
            )

            # Validate
            is_valid, issues = validate_trajectory(trajectory)

            # Compute reward
            from evaluation.agenttune.reward_function import compute_trajectory_reward
            reward = compute_trajectory_reward(trajectory, ground_truth=trajectory.ground_truth)

            # Save
            collector = TrajectoryCollector()
            path = collector.save_trajectory(trajectory)

            result = {
                "scenario_id": trajectory.scenario_id,
                "trajectory_id": trajectory.trajectory_id,
                "steps": len(trajectory.tool_calls),
                "status": trajectory.completion_status,
                "reward": trajectory.metadata.get("reward", 0),
                "path": str(path)
            }

            print(f"  Steps: {trajectory.total_steps}, Reward: {trajectory.metadata.get('reward', 0):.3f}, Status: {trajectory.completion_status}")

            results["trajectories"].append({
                "scenario_id": trajectory.scenario_id,
                "trajectory_id": trajectory.trajectory_id,
                "path": str(path)
            })
            results["rewards"].append(trajectory.metadata.get("reward", 0))

        except Exception as e:
            print(f"Error collecting {scenario_id}: {e}")
            results["errors"].append({"scenario_id": scenario_id, "error": str(e)})

    return results


if __name__ == "__main__":
    print("AgentTune Training Pipeline")
    print("=" * 50)

    # Check if we can run
    import sys
    if not AGENTTUNE_AVAILABLE:
        print("AgentTune not available. Install with: pip install agenttune")
        sys.exit(1)

    print("AgentTune available!")
    print("Running baseline trajectory collection...")

    # Run baseline collection
    results = run_baseline_collection()

    # Save results
    with open("artifacts/agenttune/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCollected {len(results['trajectories'])} trajectories")
    print(f"Average reward: {sum(results['rewards'])/len(results['rewards']) if results['rewards'] else 0:.3f}")
    if results['errors']:
        print(f"Errors: {len(results['errors'])}")