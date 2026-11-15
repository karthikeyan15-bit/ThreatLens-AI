"""AgentTune-compatible reward functions for ThreatLens investigation trajectories.

Each reward function follows the AgentTune/GRPO signature::

    def reward_fn(completions, **kwargs) -> list[float]

where ``completions`` is a list of model-generated text completions and the
function returns one score per completion. These are passed to
``agenttune.train_agentic(..., reward_funcs=[...])``.
"""

from typing import Any, Dict, List

import json


def _parse_completion(completion: Any) -> Dict[str, Any]:
    """Best-effort parse of a completion into a dict."""
    if isinstance(completion, dict):
        return completion
    if isinstance(completion, str):
        try:
            return json.loads(completion)
        except json.JSONDecodeError:
            return {"raw": completion}
    return {"raw": str(completion)}


def _extract_tool_calls(completion: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool call records from a completion dict."""
    calls = completion.get("tool_calls", [])
    if calls is None:
        calls = []
    if isinstance(calls, dict):
        calls = [calls]
    return calls


def _extract_conclusion(completion: Dict[str, Any]) -> Dict[str, Any]:
    return completion.get("conclusion", {}) or {}


def reward_tool_selection(completions, **kwargs) -> List[float]:
    """Reward for selecting the correct tools (intersection / expected)."""
    expected_list = kwargs.get("expected_tools", [])
    scores = []
    for completion in completions:
        c = _parse_completion(completion)
        actual = [tc.get("tool", "") for tc in _extract_tool_calls(c)]
        expected = expected_list[len(scores)] if len(scores) < len(expected_list) else []
        if not expected:
            scores.append(1.0 if not actual else 0.5)
            continue
        expected_set = set(expected)
        actual_set = set(actual)
        intersection = expected_set & actual_set
        scores.append(len(intersection) / len(expected_set) if expected_set else 1.0)
    return scores


def reward_argument_accuracy(completions, **kwargs) -> List[float]:
    """Reward for providing correct arguments to tools."""
    expected_calls = kwargs.get("expected_calls", [])
    scores = []
    for completion in completions:
        c = _parse_completion(completion)
        actual = _extract_tool_calls(c)
        expected = expected_calls[len(scores)] if len(scores) < len(expected_calls) else []
        if not expected:
            scores.append(1.0)
            continue
        matches = 0
        total = 0
        for exp in expected:
            exp_tool = exp.get("tool")
            exp_args = exp.get("arguments", {}) or {}
            found = False
            for act in actual:
                if act.get("tool") == exp_tool:
                    act_args = act.get("arguments", {}) or {}
                    if set(exp_args.items()).issubset(set(act_args.items())):
                        matches += 1
                    found = True
                    break
            total += 1
        scores.append(round(matches / total, 3) if total else 1.0)
    return scores


def reward_evidence_grounding(completions, **kwargs) -> List[float]:
    """Reward for conclusions grounded in confirmed evidence."""
    scores = []
    for completion in completions:
        c = _parse_completion(completion)
        evidence = c.get("evidence", []) or []
        risk = _extract_conclusion(c).get("risk_level", "")
        if not evidence:
            scores.append(0.0)
            continue
        confirmed = sum(1 for e in evidence if e.get("veracity") == "confirmed")
        total = len(evidence)
        score = confirmed / total
        # High/Critical risk requires confirmed evidence
        if risk in ("High", "Critical") and confirmed == 0:
            score = 0.0
        scores.append(score)
    return scores


def reward_task_completion(completions, **kwargs) -> List[float]:
    """Reward for completing the investigation (completed/escalated)."""
    statuses = kwargs.get("completion_status", [])
    scores = []
    for i, completion in enumerate(completions):
        c = _parse_completion(completion)
        status = c.get("completion_status")
        if status is None and i < len(statuses):
            status = statuses[i]
        if status in ("completed", "escalated"):
            scores.append(1.0)
        elif status == "max_steps":
            scores.append(0.5)
        else:
            scores.append(0.0)
    return scores


def reward_classification_accuracy(completions, **kwargs) -> List[float]:
    """Reward for correct risk classification (with partial credit)."""
    expected_risks = kwargs.get("expected_risk", [])
    risk_order = ["Inconclusive", "Low", "Medium", "High", "Critical"]
    scores = []
    for i, completion in enumerate(completions):
        c = _parse_completion(completion)
        predicted = _extract_conclusion(c).get("risk_level", "")
        expected = expected_risks[i] if i < len(expected_risks) else ""
        if not expected:
            scores.append(0.5)
            continue
        if predicted == expected:
            scores.append(1.0)
            continue
        try:
            d = abs(risk_order.index(predicted) - risk_order.index(expected))
            scores.append(max(0.0, 1.0 - d * 0.25))
        except ValueError:
            scores.append(0.0)
    return scores


def reward_escalation_accuracy(completions, **kwargs) -> List[float]:
    """Reward for correct human-escalation decision."""
    expected = kwargs.get("escalation_decision", [])
    scores = []
    for i, completion in enumerate(completions):
        c = _parse_completion(completion)
        predicted = _extract_conclusion(c).get("escalate", False)
        exp = expected[i] if i < len(expected) else False
        scores.append(1.0 if bool(predicted) == bool(exp) else 0.0)
    return scores


def reward_no_duplicate_calls(completions, **kwargs) -> List[float]:
    """Penalize duplicate tool calls with identical arguments."""
    scores = []
    for completion in completions:
        c = _parse_completion(completion)
        calls = _extract_tool_calls(c)
        seen = set()
        duplicate = False
        for tc in calls:
            key = (tc.get("tool"), json.dumps(tc.get("arguments", {}), sort_keys=True))
            if key in seen:
                duplicate = True
                break
            seen.add(key)
        scores.append(0.0 if duplicate else 1.0)
    return scores


def reward_no_unsupported_conclusions(completions, **kwargs) -> List[float]:
    """Penalize High/Critical conclusions without confirmed evidence."""
    scores = []
    for completion in completions:
        c = _parse_completion(completion)
        risk = _extract_conclusion(c).get("risk_level", "")
        evidence = c.get("evidence", []) or []
        if risk in ("High", "Critical"):
            confirmed = sum(1 for e in evidence if e.get("veracity") == "confirmed")
            scores.append(0.0 if confirmed == 0 else 1.0)
        else:
            scores.append(1.0)
    return scores


# Registry of all AgentTune reward functions by name
AGENTTUNE_REWARD_FUNCS = {
    "tool_selection": reward_tool_selection,
    "argument_accuracy": reward_argument_accuracy,
    "evidence_grounding": reward_evidence_grounding,
    "task_completion": reward_task_completion,
    "classification_accuracy": reward_classification_accuracy,
    "escalation_accuracy": reward_escalation_accuracy,
    "no_duplicate_calls": reward_no_duplicate_calls,
    "no_unsupported_conclusions": reward_no_unsupported_conclusions,
}

# Default reward weights (tunable via config)
AGENTTUNE_REWARD_WEIGHTS = {
    "tool_selection": 1.0,
    "argument_accuracy": 1.0,
    "evidence_grounding": 1.5,
    "task_completion": 1.5,
    "classification_accuracy": 2.0,
    "escalation_accuracy": 1.5,
    "no_duplicate_calls": 0.5,
    "no_unsupported_conclusions": 1.0,
}

__all__ = [
    "reward_tool_selection",
    "reward_argument_accuracy",
    "reward_evidence_grounding",
    "reward_task_completion",
    "reward_classification_accuracy",
    "reward_escalation_accuracy",
    "reward_no_duplicate_calls",
    "reward_no_unsupported_conclusions",
    "AGENTTUNE_REWARD_FUNCS",
    "AGENTTUNE_REWARD_WEIGHTS",
]
