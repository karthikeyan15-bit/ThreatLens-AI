"""Configurable model loader with explicit inference-source reporting.

The deployment Space must never silently report a heuristic result as
trained ML inference. Every prediction carries a `source` field that is
one of:

    trained_ml        - loaded from a real trained artifact
    structural        - rule-based structural analysis, no model needed
    heuristic_fallback - keyword/regex scoring, model unavailable
    model_unavailable  - artifact missing and download failed

This module also exposes actual model metrics from
models/<name>_classifier_metrics.json so the UI can show real
benchmark numbers instead of fabricated accuracy claims.
"""

import json
from pathlib import Path
from typing import Any, Optional

from config.settings import (
    HF_MODEL_REPO_ID,
    MODELS_DIR,
    MODEL_FILES,
    MODEL_METRICS_FILES,
)


class ModelLoadError(Exception):
    """Raised when a model artifact cannot be loaded or downloaded."""


class ModelUnavailableError(ModelLoadError):
    """Model artifact is missing and download failed."""


def _load_metrics(model_name: str) -> Optional[dict]:
    """Load saved model metrics JSON, or None if absent."""
    path = MODELS_DIR / MODEL_METRICS_FILES.get(model_name, "")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_model(model_name: str) -> dict:
    """Load a trained classifier artifact.

    Resolution order:
        1. Local disk (MODELS_DIR)
        2. Hugging Face Hub model repository
        3. Return model_unavailable status

    Returns a dict with:
        pipeline: the trained scikit-learn pipeline (or None)
        source: "trained_ml" | "model_unavailable"
        metrics: actual saved metrics (or None)
        error: human-readable failure reason (or None)
    """
    local_path = MODELS_DIR / MODEL_FILES.get(model_name, "")
    metrics = _load_metrics(model_name)

    if local_path.exists():
        try:
            import joblib
            pipeline = joblib.load(local_path)
            return {
                "pipeline": pipeline,
                "source": "trained_ml",
                "metrics": metrics,
                "path": str(local_path),
                "error": None,
            }
        except Exception as exc:
            raise ModelLoadError(f"Failed to load {local_path}: {exc}") from exc

    # Try downloading from Hugging Face Hub
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=HF_MODEL_REPO_ID,
            filename=MODEL_FILES.get(model_name, ""),
            repo_type="model",
        )
        import joblib
        pipeline = joblib.load(downloaded)
        return {
            "pipeline": pipeline,
            "source": "trained_ml",
            "metrics": metrics,
            "path": downloaded,
            "error": None,
        }
    except Exception as exc:
        return {
            "pipeline": None,
            "source": "model_unavailable",
            "metrics": metrics,
            "path": None,
            "error": f"Model '{model_name}' not found locally and download failed: {exc}",
        }


def format_model_status(model_name: str, result: dict) -> str:
    """Return a human-readable status string for the UI."""
    if result.get("source") == "trained_ml":
        metrics = result.get("metrics") or {}
        acc = metrics.get("test_accuracy")
        if acc:
            return f"Trained ML model loaded (test accuracy: {acc:.1%})"
        return "Trained ML model loaded"
    err = result.get("error") or "Model artifact unavailable"
    return f"Model unavailable — {err}"


def get_inference_source_label(source: str) -> str:
    """Map internal source constants to UI labels."""
    labels = {
        "trained_ml": "Trained ML classifier",
        "structural": "Structural analysis",
        "heuristic_fallback": "Heuristic fallback",
        "model_unavailable": "Model unavailable",
    }
    return labels.get(source, source)