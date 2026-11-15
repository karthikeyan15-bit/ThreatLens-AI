"""URLhaus threat intelligence client with caching, retries, and safety enforcement."""

import requests

from .base import ThreatIntelClient, Verdict


class URLHausClient(ThreatIntelClient):
    service_name = "urlhaus"
    api_url = "https://urlhaus-api.abuse.ch/v1/url/"

    def _search_impl(self, indicator: str, indicator_type: str) -> dict:
        if indicator_type != "url":
            return self._build_result({}, errors=[f"URLhaus only supports URL indicators, got {indicator_type}"],
                                     verdict=Verdict.INCONCLUSIVE)

        payload = {"url": indicator}
        resp = requests.post(self.api_url, data=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("query_status", "")
        if status == "ok" and data.get("url_info"):
            info = data["url_info"]
            return self._build_result({
                "found": True,
                "url": info.get("url"),
                "threat_type": info.get("threat"),
                "first_seen": info.get("first_seen"),
                "last_seen": info.get("last_seen"),
                "tags": info.get("tags", []),
                "payload_sha256": info.get("payload_sha256"),
            }, verdict=Verdict.MALICIOUS)

        return self._build_result({"found": False}, verdict=Verdict.INCONCLUSIVE)

    def search(self, indicator: str, indicator_type: str) -> dict:
        return self._safe_search(indicator, indicator_type)