"""The five canonical Forge charts.

Each function takes structured data (typically from the benchmark parser, the
quality-eval comparator, or the cost model), renders a chart, writes both PNG
and SVG to ``output_dir``, and returns the two paths. Chart-data preparation is
the testable surface; matplotlib's pixel output is not unit-tested.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from forge.benchmark.metrics import BenchmarkRow
from forge.cost.model import CostComparison
from forge.eval.compare import QualityDelta
from forge.plots.style import apply_style, palette


@dataclass(frozen=True)
class ChartPaths:
    """The two on-disk outputs of every chart render."""

    png: Path
    svg: Path


def _save_pair(fig: Figure, output_dir: Path, stem: str) -> ChartPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png)
    fig.savefig(svg)
    plt.close(fig)
    return ChartPaths(png=png, svg=svg)


def plot_throughput_vs_concurrency(
    *,
    series: Mapping[str, Sequence[BenchmarkRow]],
    output_dir: Path,
    stem: str = "throughput-vs-concurrency",
) -> ChartPaths:
    """Total token throughput (tok/s) as a function of concurrent requests.

    ``series`` maps a label (``"bf16"``, ``"awq"``, …) to the rows for that
    config. Rows must be ordered by concurrency.
    """
    apply_style()
    colors = palette()
    fig, ax = plt.subplots()

    for label, rows in series.items():
        color = colors.get(label, colors["muted"])
        xs = [r.concurrency for r in rows]
        ys = [r.total_token_throughput for r in rows]
        ax.plot(xs, ys, marker="o", color=color, label=label.upper())

    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Total throughput (tokens / sec)")
    ax.set_title("Throughput vs. concurrency")
    if any(r.concurrency >= 8 for rows in series.values() for r in rows):
        ax.set_xscale("log", base=2)
    ax.legend(title="Variant")
    return _save_pair(fig, output_dir, stem)


def plot_latency_vs_concurrency(
    *,
    series: Mapping[str, Sequence[BenchmarkRow]],
    metric: str,
    output_dir: Path,
    stem: str | None = None,
) -> ChartPaths:
    """Latency-distribution chart: TTFT or TPOT, p50 and p99 per series.

    ``metric`` is ``"ttft"`` or ``"tpot"``; selects which ``LatencyDistribution``
    on the rows to plot. Both percentiles are drawn per series: solid = p50,
    dashed = p99.
    """
    if metric not in ("ttft", "tpot"):
        raise ValueError(f"metric must be 'ttft' or 'tpot', got {metric!r}")

    apply_style()
    colors = palette()
    fig, ax = plt.subplots()

    for label, rows in series.items():
        color = colors.get(label, colors["muted"])
        xs = [r.concurrency for r in rows]
        latencies = [getattr(r, metric) for r in rows]
        p50 = [latency.median_ms for latency in latencies]
        p99 = [latency.p99_ms for latency in latencies]
        ax.plot(xs, p50, marker="o", color=color, label=f"{label.upper()} p50")
        ax.plot(xs, p99, marker="s", linestyle="--", color=color, label=f"{label.upper()} p99")

    metric_display = "TTFT" if metric == "ttft" else "TPOT"
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel(f"{metric_display} (ms)")
    ax.set_title(f"{metric_display} vs. concurrency")
    if any(r.concurrency >= 8 for rows in series.values() for r in rows):
        ax.set_xscale("log", base=2)
    ax.legend(title="Variant / percentile")
    return _save_pair(fig, output_dir, stem or f"{metric}-vs-concurrency")


def plot_cost_comparison(
    *,
    comparison: CostComparison,
    output_dir: Path,
    stem: str = "cost-per-1m-tokens",
) -> ChartPaths:
    """Horizontal bar chart of $/1M tokens, self-hosted vs commercial API."""
    apply_style()
    colors = palette()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    rows = list(comparison.self_hosted) + list(comparison.api)
    labels = [s.label for s in rows]
    values = [s.usd_per_1m_tokens for s in rows]
    sh_count = len(comparison.self_hosted)
    bar_colors = [
        colors["bf16"] if "bf16" in s.label.lower() else colors["awq"]
        for s in comparison.self_hosted
    ] + [
        colors["api_gpt4o"] if "gpt" in s.label.lower() else colors["api_claude"]
        for s in comparison.api
    ]

    y_positions = list(range(len(rows)))
    ax.barh(y_positions, values, color=bar_colors)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("USD per 1M tokens (lower is cheaper)")
    ax.set_title("Cost per 1M tokens — self-hosted vs. commercial APIs")

    for y, value in zip(y_positions, values, strict=True):
        ax.text(value, y, f"  ${value:,.2f}", va="center", fontsize=10)

    if 0 < sh_count < len(rows):
        ax.axhline(sh_count - 0.5, color=colors["muted"], linewidth=0.8, linestyle=":")

    return _save_pair(fig, output_dir, stem)


def plot_quality_retention(
    *,
    delta: QualityDelta,
    output_dir: Path,
    stem: str = "quality-retention",
) -> ChartPaths:
    """Per-task retention bar chart with the aggregate mean annotated."""
    apply_style()
    colors = palette()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    labels = [d.task for d in delta.per_task]
    retentions = [d.retention_pct for d in delta.per_task]

    bars = ax.bar(labels, retentions, color=colors["awq"], edgecolor="white", width=0.55)

    ax.axhline(100.0, color=colors["baseline"], linestyle="--", linewidth=1.2)
    ax.text(
        len(labels) - 0.5,
        100.0,
        " baseline (100%)",
        va="center",
        ha="left",
        fontsize=9,
        color=colors["baseline"],
    )

    for bar, value in zip(bars, retentions, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Quality retention vs. BF16 (%)")
    ax.set_ylim(min(retentions) - 5 if retentions else 0, 105)
    ax.set_title(
        f"AWQ-INT4 quality retention — mean {delta.mean_retention_pct:.1f}% vs. {delta.baseline_label.upper()}"
    )
    return _save_pair(fig, output_dir, stem)
