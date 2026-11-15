"""Risk threshold configuration for ThreatLens AI."""

# Risk level thresholds (normalized score 0.0 - 1.0)
LOW = 0.2
MEDIUM = 0.4
HIGH = 0.65
CRITICAL = 0.85

# Threshold order for iteration
THRESHOLDS = [
    ("low", LOW),
    ("medium", MEDIUM),
    ("high", HIGH),
    ("critical", CRITICAL),
]


def get_risk_level(score: float) -> str:
    """
    Convert a normalized risk score to a risk level.

    Args:
        score: Normalized risk score between 0.0 and 1.0

    Returns:
        Risk level string: "Low", "Medium", "High", "Critical", or "Inconclusive"
    """
    if score >= CRITICAL:
        return "Critical"
    elif score >= HIGH:
        return "High"
    elif score >= MEDIUM:
        return "Medium"
    elif score > 0:
        return "Low"
    else:
        return "Inconclusive"


def get_all_thresholds() -> dict:
    """Get all thresholds as a dictionary."""
    return {
        "low": LOW,
        "medium": MEDIUM,
        "high": HIGH,
        "critical": CRITICAL,
    }