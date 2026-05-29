"""Tests for the chart pipeline.

The pixel output isn't visually validated. We assert that:
- ``apply_style()`` updates rcParams without raising
- ``palette()`` returns a dict with the known semantic keys
- Each chart writes both a PNG and an SVG at the requested path
- Chart files are non-empty (matplotlib didn't silently produce a 0-byte file)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import pytest

from forge.benchmark.metrics import BenchmarkRow, LatencyDistribution, parse_result
from forge.cost.model import build_self_hosted, compare
from forge.eval.compare import QualityDelta, TaskDelta
from forge.plots.charts import (
    plot_cost_comparison,
    plot_latency_vs_concurrency,
    plot_quality_retention,
    plot_throughput_vs_concurrency,
)
from forge.plots.style import apply_style, palette

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_apply_style_updates_rcparams() -> None:
    mpl.rcParams.update({"axes.titlesize": 99})
    apply_style()
    assert mpl.rcParams["axes.titlesize"] == 13


def test_palette_has_canonical_keys() -> None:
    p = palette()
    for key in ("bf16", "awq", "api_gpt4o", "api_claude", "baseline"):
        assert key in p
        assert p[key].startswith("#")


def _rows() -> list[BenchmarkRow]:
    return [
        parse_result(FIXTURE_DIR / "bench-c0001.json", concurrency=1),
        parse_result(FIXTURE_DIR / "bench-c0002.json", concurrency=2),
    ]


def test_throughput_chart_writes_png_and_svg(tmp_path: Path) -> None:
    paths = plot_throughput_vs_concurrency(series={"bf16": _rows()}, output_dir=tmp_path, stem="th")
    assert paths.png.exists() and paths.png.stat().st_size > 1000
    assert paths.svg.exists() and paths.svg.stat().st_size > 500


def test_ttft_chart_writes_png_and_svg(tmp_path: Path) -> None:
    paths = plot_latency_vs_concurrency(
        series={"bf16": _rows()}, metric="ttft", output_dir=tmp_path
    )
    assert paths.png.exists()
    assert paths.svg.exists()
    assert "ttft" in paths.png.name


def test_tpot_chart_writes_png_and_svg(tmp_path: Path) -> None:
    paths = plot_latency_vs_concurrency(
        series={"bf16": _rows()}, metric="tpot", output_dir=tmp_path
    )
    assert paths.png.exists()
    assert paths.svg.exists()
    assert "tpot" in paths.png.name


def test_latency_chart_rejects_unknown_metric(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="metric"):
        plot_latency_vs_concurrency(series={"bf16": _rows()}, metric="bogus", output_dir=tmp_path)


def test_cost_chart_writes_png_and_svg(tmp_path: Path) -> None:
    sh = build_self_hosted(
        label="AWQ on A5000",
        gpu_tier_key="runpod-rtx-a5000-pod",
        sustained_throughput_tps=2100.0,
    )
    cmp = compare([sh], ["gpt-4o", "claude-sonnet-4-6"])
    paths = plot_cost_comparison(comparison=cmp, output_dir=tmp_path)
    assert paths.png.exists() and paths.png.stat().st_size > 1000
    assert paths.svg.exists()


def test_quality_retention_chart_writes_png_and_svg(tmp_path: Path) -> None:
    delta = QualityDelta(
        baseline_label="bf16",
        candidate_label="awq",
        per_task=[
            TaskDelta(
                task="mmlu",
                metric="acc",
                baseline_score=0.68,
                candidate_score=0.67,
                absolute_delta_pp=-1.0,
                retention_pct=98.5,
            ),
            TaskDelta(
                task="gsm8k",
                metric="exact_match",
                baseline_score=0.82,
                candidate_score=0.80,
                absolute_delta_pp=-2.0,
                retention_pct=97.6,
            ),
        ],
        mean_retention_pct=98.05,
    )
    paths = plot_quality_retention(delta=delta, output_dir=tmp_path)
    assert paths.png.exists() and paths.png.stat().st_size > 1000
    assert paths.svg.exists()


def test_throughput_chart_handles_multi_series(tmp_path: Path) -> None:
    paths = plot_throughput_vs_concurrency(
        series={"bf16": _rows(), "awq": _rows()},
        output_dir=tmp_path,
    )
    assert paths.png.exists()


def test_latency_distribution_dataclass_accessible() -> None:
    """The chart uses ``getattr(row, metric).median_ms`` — sanity check."""
    row = _rows()[0]
    assert isinstance(row.ttft, LatencyDistribution)
    assert isinstance(row.tpot, LatencyDistribution)
