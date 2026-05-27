"""Tests for lm-eval output parsing and quality-retention computation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.eval.compare import (
    QualityDelta,
    TaskResult,
    compute_deltas,
    parse_lm_eval_output,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_parse_extracts_canonical_metrics() -> None:
    rows = parse_lm_eval_output(FIXTURE_DIR / "eval-bf16.json")

    by_task = {r.task: r for r in rows}
    assert set(by_task) == {"hellaswag", "mmlu", "gsm8k"}
    assert by_task["hellaswag"].metric == "acc_norm"
    assert by_task["hellaswag"].score == pytest.approx(0.7589)
    assert by_task["mmlu"].metric == "acc"
    assert by_task["mmlu"].score == pytest.approx(0.6812)
    assert by_task["gsm8k"].metric == "exact_match"
    assert by_task["gsm8k"].score == pytest.approx(0.8245)


def test_parse_returns_empty_for_unknown_payload(tmp_path: Path) -> None:
    payload = {"results": {"some-other-task": {"acc,none": 0.5}}}
    path = tmp_path / "x.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert parse_lm_eval_output(path) == []


def test_parse_raises_on_top_level_not_object(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="expected JSON object"):
        parse_lm_eval_output(path)


def test_parse_raises_on_missing_results(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(KeyError, match="results"):
        parse_lm_eval_output(path)


def test_compute_deltas_against_real_fixtures() -> None:
    baseline = parse_lm_eval_output(FIXTURE_DIR / "eval-bf16.json")
    candidate = parse_lm_eval_output(FIXTURE_DIR / "eval-awq.json")

    delta = compute_deltas(baseline, candidate, baseline_label="bf16", candidate_label="awq")

    assert isinstance(delta, QualityDelta)
    assert delta.baseline_label == "bf16"
    assert delta.candidate_label == "awq"
    assert {d.task for d in delta.per_task} == {"hellaswag", "mmlu", "gsm8k"}
    assert 97.0 <= delta.mean_retention_pct <= 100.0
    for per in delta.per_task:
        assert per.retention_pct < 100.0


def test_compute_deltas_handles_extra_task_in_baseline_only() -> None:
    baseline = [
        TaskResult(task="mmlu", metric="acc", score=0.5),
        TaskResult(task="extra", metric="acc", score=0.4),
    ]
    candidate = [TaskResult(task="mmlu", metric="acc", score=0.49)]
    delta = compute_deltas(baseline, candidate)
    assert {d.task for d in delta.per_task} == {"mmlu"}
    assert any("extra" in note for note in delta.notes)


def test_compute_deltas_skips_zero_baseline() -> None:
    baseline = [TaskResult(task="mmlu", metric="acc", score=0.0)]
    candidate = [TaskResult(task="mmlu", metric="acc", score=0.1)]
    delta = compute_deltas(baseline, candidate)
    assert delta.per_task == []
    assert delta.mean_retention_pct == 0.0


def test_to_dict_round_trips_through_json() -> None:
    baseline = parse_lm_eval_output(FIXTURE_DIR / "eval-bf16.json")
    candidate = parse_lm_eval_output(FIXTURE_DIR / "eval-awq.json")
    delta = compute_deltas(baseline, candidate)
    payload = delta.to_dict()
    encoded = json.dumps(payload)
    assert "mean_retention_pct" in encoded
    assert "per_task" in encoded


def test_extract_score_prefers_strict_match_for_gsm8k(tmp_path: Path) -> None:
    """GSM8K must always pick ``strict-match`` regardless of dict key order.

    lm-eval emits both ``exact_match,strict-match`` and
    ``exact_match,flexible-extract``. The retention chart must not flip between
    them because the JSON happened to serialize one key first.
    """
    payload = {
        "results": {
            "gsm8k": {
                "exact_match,flexible-extract": 0.95,
                "exact_match,strict-match": 0.9,
            }
        }
    }
    path = tmp_path / "x.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rows = parse_lm_eval_output(path)
    assert len(rows) == 1
    assert rows[0].score == pytest.approx(0.9)


def test_extract_score_falls_back_to_flexible_extract(tmp_path: Path) -> None:
    """If only ``flexible-extract`` is present, it's used as a fallback."""
    payload = {
        "results": {
            "gsm8k": {
                "exact_match,flexible-extract": 0.77,
            }
        }
    }
    path = tmp_path / "x.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rows = parse_lm_eval_output(path)
    assert len(rows) == 1
    assert rows[0].score == pytest.approx(0.77)


def test_extract_score_ignores_unfiltered_keys(tmp_path: Path) -> None:
    """Bare ``acc`` (without the ``,filter`` suffix) is not an accepted key.

    lm-eval always emits ``<metric>,<filter>``; defending against the bare form
    would mask schema drift. If lm-eval ever stops emitting the comma form, we
    want that failure to be loud.
    """
    payload = {
        "results": {
            "mmlu": {"acc": 0.55},
        }
    }
    path = tmp_path / "x.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert parse_lm_eval_output(path) == []
