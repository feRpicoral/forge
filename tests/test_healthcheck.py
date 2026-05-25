"""Tests for the healthcheck URL derivation logic.

The blocking ``wait_for_healthy`` loop is not unit-tested — it makes real HTTP
calls and is exercised end-to-end during smoke runs. The pure functions around
it (URL derivation) are tested here.
"""

from __future__ import annotations

import pytest

from forge.serving.healthcheck import HealthcheckResult, _derive_health_url


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://localhost:8000/v1", "http://localhost:8000/health"),
        ("http://localhost:8000/v1/", "http://localhost:8000/health"),
        ("http://localhost:8000", "http://localhost:8000/health"),
        ("https://vllm.internal:443/v1", "https://vllm.internal:443/health"),
        ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/health"),
    ],
)
def test_derive_health_url(base_url: str, expected: str) -> None:
    assert _derive_health_url(base_url) == expected


def test_healthcheck_result_dataclass() -> None:
    result = HealthcheckResult(healthy=True, elapsed_seconds=1.5, attempts=3)
    assert result.healthy
    assert result.elapsed_seconds == pytest.approx(1.5)
    assert result.attempts == 3
    assert result.last_error is None
