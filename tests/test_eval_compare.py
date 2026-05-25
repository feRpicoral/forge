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
    # HellaSwag uses acc_norm, MMLU uses acc, GSM8K uses exact_match.
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
    # AWQ should retain ~98%+ across the board on Llama 3.1 8B.
    assert 97.0 <= delta.mean_retention_pct <= 100.0
    # Each task is < 100 (slight loss expected).
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
    # division-by-zero defenders skip, leaving the per-task list empty.
    assert delta.per_task == []
    assert delta.mean_retention_pct == 0.0


def test_to_dict_round_trips_through_json() -> None:
    baseline = parse_lm_eval_output(FIXTURE_DIR / "eval-bf16.json")
    candidate = parse_lm_eval_output(FIXTURE_DIR / "eval-awq.json")
    delta = compute_deltas(baseline, candidate)
    payload = delta.to_dict()
    # No round-trip via from_dict — just ensure serialization is clean.
    encoded = json.dumps(payload)
    assert "mean_retention_pct" in encoded
    assert "per_task" in encoded


def test_extract_score_finds_keys_with_filter_suffix(tmp_path: Path) -> None:
    """lm-eval uses ``metric,filter`` keys; parser must find the metric by prefix."""
    payload = {
        "results": {
            "gsm8k": {
                "exact_match,strict-match": 0.9,
                "exact_match,flexible-extract": 0.95,
            }
        }
    }
    path = tmp_path / "x.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rows = parse_lm_eval_output(path)
    assert len(rows) == 1
    # Strict-match comes first in the dict; the parser picks the first match.
    assert rows[0].score == pytest.approx(0.9)
