"""Risk assessment service.

Computes a normalized risk score from evidence quality and confidence,
then maps it to a risk level using configurable thresholds.

Safety rule: NEVER declare an indicator safe simply because no threat
intel provider returned a match. If all sources are inconclusive, the
risk level MUST be "Inconclusive" — never "Low".
"""

from config.settings import RISK_THRESHOLDS


def compute_risk_score(evidence_items: list[dict]) -> float:
    """Compute a normalized risk score from 0.0 to 1.0."""
    weights = {"confirmed": 1.0, "suggested": 0.5, "unknown": 0.2}
    score = 0.0
    for item in evidence_items:
        veracity = item.get("veracity", "unknown")
        confidence = item.get("confidence", 0.5)
        score += weights.get(veracity, 0.2) * confidence

    max_score = len(evidence_items) if evidence_items else 1
    normalized = min(1.0, score / max(max_score * 0.7, 1))
    return round(normalized, 3)


def risk_level(score: float) -> str:
    """Map a normalized score to a risk level."""
    if score >= RISK_THRESHOLDS["CRITICAL"]:
        return "Critical"
    if score >= RISK_THRESHOLDS["HIGH"]:
        return "High"
    if score >= RISK_THRESHOLDS["MEDIUM"]:
        return "Medium"
    if score >= RISK_THRESHOLDS["LOW"]:
        return "Low"
    return "Inconclusive"


def is_inconclusive(evidence_items: list[dict]) -> bool:
    """Return True if no evidence item is confirmed and no definitive
    threat signal exists. This enforces the safety rule that absence
    of threat intel matches is NOT a clean verdict."""
    if not evidence_items:
        return True
    confirmed = [e for e in evidence_items if e.get("veracity") == "confirmed"]
    return len(confirmed) == 0


def assess(evidence_items: list[dict]) -> dict:
    """Assess risk from evidence items.

    Returns a dict with keys:
        score, risk_level, confidence, is_inconclusive, escalate
    """
    score = compute_risk_score(evidence_items)
    level = risk_level(score)
    inconclusive = is_inconclusive(evidence_items)

    # If no confirmed evidence exists, force Inconclusive regardless of score
    if inconclusive:
        level = "Inconclusive"
        score = min(score, RISK_THRESHOLDS["LOW"] - 0.01)

    confidence = max(score, 1.0 - score) if evidence_items else 0.1
    escalate = level in ("High", "Critical", "Inconclusive")

    return {
        "score": score,
        "risk_level": level,
        "confidence": round(confidence, 3),
        "is_inconclusive": inconclusive,
        "escalate": escalate,
    }