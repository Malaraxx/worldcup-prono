"""
Session HTTP avec cache 24h, rate limit par domaine, retry exponentiel.
Retourne None en cas d'échec persistant (ne crash jamais).
"""

import hashlib
import logging
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parents[2] / "data" / "cache"
CACHE_TTL_SECONDS = 86_400  # 24h
USER_AGENT = "Mozilla/5.0 (compatible; WCPronoBot/1.0)"
RATE_LIMIT_CALLS = 1
RATE_LIMIT_PERIOD = 4  # seconds — increased to reduce 429s from Wikipedia API


def _cache_path(url: str) -> Path:
    domain = urlparse(url).netloc.replace(":", "_")
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    return CACHE_DIR / domain / f"{url_hash}.html"


def _load_cache(url: str) -> str | None:
    path = _cache_path(url)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > CACHE_TTL_SECONDS:
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _save_cache(url: str, content: str) -> None:
    path = _cache_path(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class CachedSession:
    """
    Wrapper requests.Session avec :
    - User-Agent fixe
    - Cache HTML 24h sur disque (data/cache/{domain}/{sha256}.html)
    - Rate limit 1 req / 2 sec par appel (global, pas par domaine)
    - Retry 3x backoff exponentiel sur erreurs réseau
    - Retourne None si 4xx/5xx persistant
    """

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

    @sleep_and_retry
    @limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
    def _rate_limited_get(self, url: str, **kwargs) -> requests.Response:
        return self._session.get(url, timeout=30, **kwargs)

    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=16),
        reraise=False,
    )
    def _fetch_with_retry(self, url: str, **kwargs) -> requests.Response | None:
        return self._rate_limited_get(url, **kwargs)

    def get(self, url: str, use_cache: bool = True, **kwargs) -> str | None:
        """
        GET url → contenu HTML (str) ou None si échec.
        use_cache=False force le re-fetch même si cache valide.
        """
        if use_cache:
            cached = _load_cache(url)
            if cached is not None:
                logger.debug("Cache hit: %s", url)
                return cached

        try:
            resp = self._fetch_with_retry(url, **kwargs)
        except Exception as exc:
            logger.warning("Fetch failed after retries: %s — %s", url, exc)
            return None

        if resp is None:
            logger.warning("Fetch returned None: %s", url)
            return None

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 10))
            logger.warning("HTTP 429 for %s — attente %ds", url, retry_after)
            time.sleep(retry_after)
            try:
                resp = self._fetch_with_retry(url, **kwargs)
            except Exception as exc:
                logger.warning("Fetch failed after 429 retry: %s — %s", url, exc)
                return None
            if resp is None or resp.status_code >= 400:
                logger.warning("Échec après 429 retry: %s", url)
                return None

        if resp.status_code >= 400:
            logger.warning("HTTP %s for %s", resp.status_code, url)
            return None

        content = resp.text
        if use_cache:
            _save_cache(url, content)
        return content
