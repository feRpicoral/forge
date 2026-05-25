"""Lightweight HTTP health probe for the vLLM server.

Polls ``GET /health`` with bounded retries. Uses ``urllib`` from the stdlib to
avoid a runtime dependency on ``httpx`` — this module is imported by ``scripts/``
that may run in minimal environments.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class HealthcheckResult:
    healthy: bool
    elapsed_seconds: float
    attempts: int
    last_error: str | None = None


def wait_for_healthy(
    base_url: str,
    timeout_seconds: float = 120.0,
    interval_seconds: float = 1.0,
) -> HealthcheckResult:
    """Block until ``GET {base_url}/health`` returns 200 or the timeout elapses.

    ``base_url`` should be the vLLM OpenAI-compatible base — e.g.
    ``http://localhost:8000/v1``. The ``/health`` endpoint lives one level up,
    at ``http://localhost:8000/health``, so we derive it from the base URL.
    """
    health_url = _derive_health_url(base_url)
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    last_error: str | None = None
    started = time.monotonic()

    while time.monotonic() < deadline:
        attempts += 1
        try:
            with urllib.request.urlopen(health_url, timeout=interval_seconds) as resp:
                if 200 <= resp.status < 300:
                    return HealthcheckResult(
                        healthy=True,
                        elapsed_seconds=time.monotonic() - started,
                        attempts=attempts,
                    )
                last_error = f"HTTP {resp.status}"
        except urllib.error.URLError as exc:
            last_error = str(exc.reason)
        except TimeoutError as exc:
            last_error = f"timeout: {exc}"

        time.sleep(interval_seconds)

    return HealthcheckResult(
        healthy=False,
        elapsed_seconds=time.monotonic() - started,
        attempts=attempts,
        last_error=last_error,
    )


def _derive_health_url(base_url: str) -> str:
    """Strip ``/v1`` suffix if present and append ``/health``."""
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    return f"{trimmed}/health"
