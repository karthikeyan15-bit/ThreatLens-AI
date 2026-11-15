"""Centralized configuration for ThreatLens AI.

All runtime settings live here so the application can start without
optional API keys and degrade gracefully when model artifacts or
external services are unavailable.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATABASE_PATH = os.getenv("DATABASE_PATH", str(PROJECT_ROOT / "database" / "threatlens.db"))
REPORTS_DIR = PROJECT_ROOT / "reports"

# ─── Hugging Face Router (active LLM provider) ────────────────────────────
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "CohereLabs/tiny-aya-global:cohere")
HF_PROVIDER = os.getenv("HF_PROVIDER", "cohere")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Hugging Face Inference Endpoints base URL (OpenAI-compatible)
HF_BASE_URL = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")

# Default generation parameters
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.2"))
HF_MAX_TOKENS = int(os.getenv("HF_MAX_TOKENS", "2048"))

# Agent version → model mapping for baseline vs improved evaluation.
# NOTE: Both baseline and improved currently use the same HF_MODEL_ID because
# separate model variants are not independently available/tested on the HF Router.
# AgentTune fine-tuning has NOT been completed. Do not claim performance
# improvement from these configurations.
AGENT_MODELS = {
    "baseline": HF_MODEL_ID,
    "improved": HF_MODEL_ID,
}

# ─── ML model artifacts ───────────────────────────────────────────────────
# Model artifacts live in a dedicated Hugging Face Model repository.
# The deployment Space does not ship trained weights.
HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "karthikeyan1509/threatlens-models")
MODEL_FILES = {
    "email": "email_classifier.joblib",
    "url": "url_classifier.joblib",
    "network": "network_classifier.joblib",
}
MODEL_METRICS_FILES = {
    "email": "email_classifier_metrics.json",
    "url": "url_classifier_metrics.json",
    "network": "network_classifier_metrics.json",
}

# ─── Threat intelligence ──────────────────────────────────────────────────
THREAT_INTEL_KEYS = {
    "urlhaus": os.getenv("URLHAUS_API_KEY", ""),
    "openphish": os.getenv("OPENPHISH_API_KEY", ""),  # OpenPhish Community Feed requires no key
    "abuseipdb": os.getenv("ABUSEIPDB_API_KEY", ""),
    "virustotal": os.getenv("VIRUSTOTAL_API_KEY", ""),
}

# ─── Investigation limits ─────────────────────────────────────────────────
MAX_INVESTIGATION_STEPS = int(os.getenv("MAX_INVESTIGATION_STEPS", "12"))
TOOL_TIMEOUT_SECONDS = int(os.getenv("TOOL_TIMEOUT_SECONDS", "10"))
THREAT_INTEL_TIMEOUT_SECONDS = int(os.getenv("THREAT_INTEL_TIMEOUT_SECONDS", "8"))

# ─── Risk thresholds ──────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "LOW": 0.2,
    "MEDIUM": 0.4,
    "HIGH": 0.65,
    "CRITICAL": 0.85,
}

# ─── Database persistence note ────────────────────────────────────────────
# Hugging Face Spaces storage is ephemeral unless a persistent volume is
# configured in the Space settings. Investigation history may be lost on
# restart unless persistence is enabled.
HF_PERSISTENT_STORAGE_ENV = "HF_PERSISTENT_STORAGE_PATH"