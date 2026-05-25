"""Smoke test — the package imports and exposes a version."""

from __future__ import annotations

import forge


def test_version_is_set() -> None:
    assert isinstance(forge.__version__, str)
    assert forge.__version__.count(".") == 2
