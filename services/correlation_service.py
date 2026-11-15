"""Indicator correlation service.

Finds relationships between evidence items using deterministic rules
plus optional LLM reasoning. Never treats a lack of threat-intel match
as a clean result.
"""

import re
from typing import Any, Optional


def extract_domain(value: str) -> Optional[str]:
    """Extract the registered domain from a URL, IP, or domain string."""
    if not value:
        return None
    value = value.strip()
    if "://" in value:
        value = value.split("://", 1)[1]
    value = value.split("/")[0].split("?")[0].split("#")[0]
    if re.match(r"^\d+\.\d+\.\d+\.\d+$", value):
        return None
    parts = value.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return None


def deterministic_correlations(evidence_items: list[dict]) -> list[dict]:
    """Find correlations using deterministic rules only.

    Returns a list of correlation dicts with keys:
        indicator_a, indicator_b, relationship, strength
    """
    correlations = []
    seen_pairs = set()

    for i, a in enumerate(evidence_items):
        for b in evidence_items[i + 1:]:
            domain_a = extract_domain(a.get("indicator_value", ""))
            domain_b = extract_domain(b.get("indicator_value", ""))
            ip_a = a.get("indicator_value", "")
            ip_b = b.get("indicator_value", "")

            # Same registered domain
            if domain_a and domain_b and domain_a == domain_b:
                key = tuple(sorted([a["indicator_value"], b["indicator_value"]]))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    correlations.append({
                        "indicator_a": a["indicator_value"],
                        "indicator_b": b["indicator_value"],
                        "relationship": "same_domain",
                        "strength": 0.9,
                    })

            # Same IP
            if ip_a and ip_b and ip_a == ip_b and a["indicator_value"] != b["indicator_value"]:
                key = tuple(sorted([a["indicator_value"], b["indicator_value"]]))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    correlations.append({
                        "indicator_a": a["indicator_value"],
                        "indicator_b": b["indicator_value"],
                        "relationship": "same_ip",
                        "strength": 0.95,
                    })

            # Email sender domain matches URL domain
            if a.get("indicator_type") == "email" and domain_b:
                email_domain = a["indicator_value"].split("@")[-1] if "@" in a["indicator_value"] else None
                if email_domain and email_domain == domain_b:
                    key = tuple(sorted([a["indicator_value"], b["indicator_value"]]))
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        correlations.append({
                            "indicator_a": a["indicator_value"],
                            "indicator_b": b["indicator_value"],
                            "relationship": "email_to_domain",
                            "strength": 0.7,
                        })

    return correlations


def llm_correlations(evidence_items: list[dict], llm_client: Any) -> list[dict]:
    """Use the LLM to find additional correlations between evidence items.

    Returns an empty list if the LLM is unavailable or returns invalid JSON.
    """
    if len(evidence_items) < 2:
        return []

    items_text = "\n".join(
        f"- [{e.get('indicator_type')}] {e.get('indicator_value')} "
        f"(veracity: {e.get('veracity')})"
        for e in evidence_items
    )

    prompt = f"""Given these investigation evidence items, identify any correlations or relationships between them.
Return a JSON array like: [{{"indicator_a": "...", "indicator_b": "...", "relationship": "...", "strength": 0.0-1.0}}]

Evidence:
{items_text}

If no clear correlations exist, return an empty array."""

    try:
        response = llm_client.messages_create(
            model="tiny-aya-global",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else "[]"
        import json
        return json.loads(text.strip())
    except Exception:
        return []


def correlate(evidence_items: list[dict], llm_client: Optional[Any] = None) -> list[dict]:
    """Run deterministic correlation, optionally augmented by LLM reasoning."""
    correlations = deterministic_correlations(evidence_items)
    if llm_client is not None:
        correlations.extend(llm_correlations(evidence_items, llm_client))
    return correlations