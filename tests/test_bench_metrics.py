"""Tests for the ``vllm bench serve`` JSON parser.

Uses a golden fixture captured from a real ``vllm bench serve`` run on M1 CPU
against ``Qwen/Qwen2.5-0.5B-Instruct``. If vLLM ever changes the output schema
in a way that breaks the parser, these tests fail loud — that is the design.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from forge.benchmark.metrics import (
    BenchmarkRow,
    LatencyDistribution,
    load_results,
    parse_result,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_parse_real_vllm_output() -> None:
    row = parse_result(FIXTURE_DIR / "bench-c0002.json", concurrency=2)

    assert isinstance(row, BenchmarkRow)
    assert row.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert row.backend == "openai-chat"
    assert row.num_prompts == 8
    assert row.concurrency == 2
    assert row.duration_seconds == pytest.approx(6.65, abs=0.01)
    assert row.request_throughput == pytest.approx(1.20, abs=0.01)
    assert row.output_throughput == pytest.approx(60.43, abs=0.01)
    assert row.total_token_throughput == pytest.approx(211.37, abs=0.01)


def test_parse_extracts_ttft_distribution() -> None:
    row = parse_result(FIXTURE_DIR / "bench-c0002.json", concurrency=2)

    assert isinstance(row.ttft, LatencyDistribution)
    assert row.ttft.mean_ms == pytest.approx(86.43, abs=0.01)
    assert row.ttft.median_ms == pytest.approx(77.69, abs=0.01)
    assert row.ttft.p99_ms == pytest.approx(146.01, abs=0.01)
    assert row.ttft.std_ms is not None
    assert row.ttft.std_ms == pytest.approx(24.69, abs=0.01)


def test_parse_extracts_tpot_distribution() -> None:
    row = parse_result(FIXTURE_DIR / "bench-c0002.json", concurrency=2)

    assert isinstance(row.tpot, LatencyDistribution)
    assert row.tpot.mean_ms == pytest.approx(30.93, abs=0.01)
    assert row.tpot.median_ms == pytest.approx(28.44, abs=0.01)
    assert row.tpot.p99_ms == pytest.approx(49.14, abs=0.01)


def test_parse_extracts_optional_itl() -> None:
    row = parse_result(FIXTURE_DIR / "bench-c0002.json", concurrency=2)

    assert row.itl is not None
    assert row.itl.mean_ms == pytest.approx(27.61, abs=0.01)


def test_parse_dataset_name_missing_falls_back_to_unknown() -> None:
    """vLLM doesn't echo dataset_name in the result JSON — parser fills the gap."""
    row = parse_result(FIXTURE_DIR / "bench-c0002.json", concurrency=2)
    assert row.dataset == "unknown"


def test_parse_source_path_recorded() -> None:
    row = parse_result(FIXTURE_DIR / "bench-c0002.json", concurrency=2)
    assert "bench-c0002.json" in row.source_path


def test_to_dict_is_serializable() -> None:
    row = parse_result(FIXTURE_DIR / "bench-c0002.json", concurrency=2)
    payload = row.to_dict()
    json.dumps(payload)
    assert payload["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert payload["concurrency"] == 2


def test_parse_raises_on_missing_required_field(tmp_path: Path) -> None:
    """Dropping a required field surfaces as KeyError, not silent zero."""
    payload = json.loads((FIXTURE_DIR / "bench-c0002.json").read_text())
    del payload["request_throughput"]
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(KeyError, match="request_throughput"):
        parse_result(broken, concurrency=2)


def test_parse_raises_on_missing_ttft(tmp_path: Path) -> None:
    payload = json.loads((FIXTURE_DIR / "bench-c0002.json").read_text())
    del payload["mean_ttft_ms"]
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(KeyError, match="mean_ttft_ms"):
        parse_result(broken, concurrency=2)


def test_parse_optional_itl_can_be_absent(tmp_path: Path) -> None:
    payload = json.loads((FIXTURE_DIR / "bench-c0002.json").read_text())
    for key in ("mean_itl_ms", "median_itl_ms", "p99_itl_ms", "std_itl_ms"):
        payload.pop(key, None)
    no_itl = tmp_path / "no-itl.json"
    no_itl.write_text(json.dumps(payload), encoding="utf-8")

    row = parse_result(no_itl, concurrency=2)
    assert row.itl is None


def test_parse_top_level_must_be_object(tmp_path: Path) -> None:
    broken = tmp_path / "list.json"
    broken.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="expected JSON object at top level"):
        parse_result(broken, concurrency=1)


def test_load_results_ordered_by_concurrency() -> None:
    rows = load_results(FIXTURE_DIR, [1, 2], name="bench")
    assert [r.concurrency for r in rows] == [1, 2]
    # Throughput should be non-decreasing as concurrency increases on a model
    # that has any batching headroom — true here.
    assert rows[1].total_token_throughput >= rows[0].total_token_throughput


def test_load_results_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path, [1], "missing")


def _fixture_payload() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((FIXTURE_DIR / "bench-c0002.json").read_text())
    return payload


def test_fixture_is_well_formed() -> None:
    """Sanity check on the fixture itself — guards against accidental edits."""
    payload = _fixture_payload()
    for required in [
        "model_id",
        "backend",
        "num_prompts",
        "duration",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
    ]:
        assert required in payload, f"fixture lost {required}"
