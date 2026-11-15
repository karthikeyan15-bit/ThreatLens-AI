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

# ─── Tiny Aya / Cohere V2 ────────────────────────────────────────────────
TINY_AYA_API_KEY = os.getenv("TINY_AYA_API_KEY", "")
TINY_AYA_BASE_URL = os.getenv("TINY_AYA_BASE_URL", "https://api.cohere.com/v2")
TINY_AYA_MODEL = os.getenv("TINY_AYA_MODEL", "tiny-aya-global")
TINY_AYA_TEMPERATURE = float(os.getenv("TINY_AYA_TEMPERATURE", "0.2"))
TINY_AYA_MAX_TOKENS = int(os.getenv("TINY_AYA_MAX_TOKENS", "2048"))

# Tiny Aya model variants verified against the Cohere V2 API.
# Only model IDs that Cohere actually serves should be listed here.
TINY_AYA_MODELS = {
    "tiny-aya-global": "Tiny Aya Global — General purpose",
    "tiny-aya-fire": "Tiny Aya Fire — High creativity",
    "tiny-aya-earth": "Tiny Aya Earth — Grounded reasoning",
    "tiny-aya-water": "Tiny Aya Water — Fluid adaptability",
}

# Agent version → model mapping for baseline vs improved evaluation.
AGENT_MODELS = {
    "baseline": "tiny-aya-global",
    "improved": "tiny-aya-earth",
}

# ─── ML model artifacts ───────────────────────────────────────────────────
# Model artifacts live in a dedicated Hugging Face Model repository.
# The deployment Space does not ship trained weights.
HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "karthikeyan1509/threatlens-models")
MODEL_FILES = {
    "email": "models/email_classifier.joblib",
    "url": "models/url_classifier.joblib",
    "network": "models/network_classifier.joblib",
}
MODEL_METRICS_FILES = {
    "email": "models/email_classifier_metrics.json",
    "url": "models/url_classifier_metrics.json",
    "network": "models/network_classifier_metrics.json",
}

# ─── Threat intelligence ──────────────────────────────────────────────────
THREAT_INTEL_KEYS = {
    "urlhaus": os.getenv("URLHAUS_API_KEY", ""),
    "phishtank": os.getenv("PHishtank_API_KEY", ""),
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