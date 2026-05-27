"""Tests for prompt-length statistics."""

from __future__ import annotations

import pytest

from forge.benchmark.workloads import LengthSummary


def test_from_values_empty_is_none() -> None:
    assert LengthSummary.from_values([]) is None


def test_from_values_single_value() -> None:
    summary = LengthSummary.from_values([42])
    assert summary is not None
    assert summary.count == 1
    assert summary.mean == 42
    assert summary.median == 42
    assert summary.p99 == 42
    assert summary.stdev == 0.0
    assert summary.min_value == 42
    assert summary.max_value == 42


def test_from_values_typical_distribution() -> None:
    values = list(range(1, 101))
    summary = LengthSummary.from_values(values)
    assert summary is not None
    assert summary.count == 100
    assert summary.mean == pytest.approx(50.5)
    assert summary.median == pytest.approx(50.5)
    # p99 of 1..100 at the [0.99 * 99] = 98 index → value 99.
    assert summary.p99 == pytest.approx(99.0)
    assert summary.min_value == 1
    assert summary.max_value == 100
    assert summary.stdev > 0


def test_from_values_unsorted_input_handled() -> None:
    values = [5, 1, 4, 2, 3]
    summary = LengthSummary.from_values(values)
    assert summary is not None
    assert summary.min_value == 1
    assert summary.max_value == 5
    assert summary.median == pytest.approx(3.0)
