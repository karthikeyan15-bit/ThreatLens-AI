"""Model configuration for ThreatLens AI."""

# Baseline model (current)
BASELINE_MODEL = "tiny-aya-global"

# Improved model (after AgentTune fine-tuning)
IMPROVED_MODEL = "tiny-aya-global-improved"

# Default model to use
DEFAULT_MODEL = BASELINE_MODEL

# Model parameters
MODEL_PARAMS = {
    "max_tokens": 1024,
    "temperature": 0.1,
    "top_p": 0.9,
}