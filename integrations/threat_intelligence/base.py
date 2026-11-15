"""Base class for all threat intelligence clients with retries, caching, and safety enforcement."""

import os
import time
import json
import sqlite3
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from pathlib import Path
from functools import wraps


CACHE_DB = Path(__file__).parent.parent.parent / "database" / "threat_intel_cache.db"
CACHE_DB.parent.mkdir(parents=True, exist_ok=True)


def init_cache_db():
    """Initialize the threat intelligence cache database."""
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS intel_cache (
                cache_key TEXT PRIMARY KEY,
                service TEXT NOT NULL,
                indicator TEXT NOT NULL,
                indicator_type TEXT NOT NULL,
                result TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_lookup ON intel_cache(service, indicator, indicator_type)")


init_cache_db()


def cache_key(service: str, indicator: str, indicator_type: str) -> str:
    """Generate cache key from service, indicator, and type."""
    return hashlib.sha256(f"{service}:{indicator_type}:{indicator}".encode()).hexdigest()[:32]


def get_cached_result(service: str, indicator: str, indicator_type: str, ttl_hours: int = 24) -> Optional[dict]:
    """Retrieve cached result if valid."""
    key = cache_key(service, indicator, indicator_type)
    with sqlite3.connect(CACHE_DB) as conn:
        row = conn.execute(
            "SELECT result FROM intel_cache WHERE cache_key = ? AND expires_at > datetime('now')",
            (key,)
        ).fetchone()
        if row:
            return json.loads(row[0])
    return None


def set_cached_result(service: str, indicator: str, indicator_type: str, result: dict, ttl_hours: int = 24):
    """Store result in cache."""
    key = cache_key(service, indicator, indicator_type)
    expires = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO intel_cache (cache_key, service, indicator, indicator_type, result, expires_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (key, service, indicator, indicator_type, json.dumps(result), expires.isoformat())
        )


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
    """Decorator for retry with exponential backoff and jitter."""
    import random

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


class Verdict:
    """Standardized verdict enum for threat intelligence."""
    MALICIOUS = "malicious"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"
    NOT_CONFIGURED = "not_configured"

    @classmethod
    def is_definitively_malicious(cls, verdict: str) -> bool:
        return verdict == cls.MALICIOUS

    @classmethod
    def is_safe(cls, verdict: str) -> bool:
        # NEVER return True for safety - the rule is "never declare safe"
        return False


class ThreatIntelClient(ABC):
    """Abstract base for threat intelligence API clients with retries, caching, and safety enforcement."""

    # Override in subclasses
    DEFAULT_TTL_HOURS = 24
    MAX_RETRIES = 3

    @abstractmethod
    def _search_impl(self, indicator: str, indicator_type: str) -> dict[str, Any]:
        """Implementation-specific search. Should return dict with 'verdict' and 'data'."""
        ...

    @abstractmethod
    def search(self, indicator: str, indicator_type: str) -> dict[str, Any]:
        """Public search method with caching, retries, and safety enforcement."""
        ...

    @property
    @abstractmethod
    def service_name(self) -> str:
        ...

    def _get_api_key(self) -> Optional[str]:
        env_var = f"{self.service_name.upper()}_API_KEY"
        return os.environ.get(env_var, "") or None

    def is_configured(self) -> bool:
        return bool(self._get_api_key())

    def _build_result(self, data: dict, errors: list[str] | None = None, verdict: str = Verdict.INCONCLUSIVE) -> dict:
        return {
            "service": self.service_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
            "errors": errors or [],
            "configured": self.is_configured(),
            "verdict": verdict,
        }

    def _safe_search(self, indicator: str, indicator_type: str) -> dict:
        """
        Perform search with caching, retries, and safety enforcement.
        NEVER returns a 'safe' verdict - only MALICIOUS, INCONCLUSIVE, ERROR, or NOT_CONFIGURED.
        """
        # Check cache first
        cached = get_cached_result(self.service_name, indicator, indicator_type, self.DEFAULT_TTL_HOURS)
        if cached:
            cached["cached"] = True
            return cached

        if not self.is_configured():
            result = self._build_result(
                {}, errors=["API key not configured"], verdict=Verdict.NOT_CONFIGURED
            )
            return result

        # Perform search with retries
        try:
            result = self._search_with_retries(indicator, indicator_type)
            # Cache successful results
            set_cached_result(self.service_name, indicator, indicator_type, result, self.DEFAULT_TTL_HOURS)
            return result
        except Exception as e:
            result = self._build_result({}, errors=[str(e)], verdict=Verdict.ERROR)
            return result

    def _search_with_retries(self, indicator: str, indicator_type: str) -> dict:
        """Execute search with retry logic."""
        last_exception = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return self._search_impl(indicator, indicator_type)
            except Exception as e:
                last_exception = e
                if attempt < self.MAX_RETRIES - 1:
                    delay = min(1.0 * (2 ** attempt), 30.0)
                    time.sleep(delay)
        raise last_exception