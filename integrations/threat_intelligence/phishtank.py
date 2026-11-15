"""PhishTank threat intelligence client (Developer API) with caching, retries, and safety enforcement."""

import requests

from .base import ThreatIntelClient, Verdict


class PhishTankClient(ThreatIntelClient):
    service_name = "phishtank"
    # Note: PhishTank public API is deprecated. Use Developer API with key.
    # For now, we mark as requiring auth and gracefully skip if not configured.

    def _search_impl(self, indicator: str, indicator_type: str) -> dict:
        if indicator_type not in ("url", "domain"):
            return self._build_result({}, errors=[f"PhishTank only supports URL/domain indicators, got {indicator_type}"],
                                     verdict=Verdict.INCONCLUSIVE)

        # Extract domain from URL if needed
        if indicator_type == "url":
            from urllib.parse import urlparse
            domain = urlparse(indicator).netloc
        else:
            domain = indicator

        # Note: PhishTank public API is deprecated (2022). This is a placeholder for Developer API.
        # Developer API requires key and uses different endpoint.
        return self._build_result(
            {"domain": domain, "note": "PhishTank public API deprecated; use Developer API with key"},
            verdict=Verdict.INCONCLUSIVE
        )

    def search(self, indicator: str, indicator_type: str) -> dict:
        return self._safe_search(indicator, indicator_type)