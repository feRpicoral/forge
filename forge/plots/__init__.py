"""Chart generation — opinionated matplotlib stylesheet + the five canonical charts."""

from forge.plots.charts import (
    plot_cost_comparison,
    plot_latency_vs_concurrency,
    plot_quality_retention,
    plot_throughput_vs_concurrency,
)
from forge.plots.style import apply_style, palette

__all__ = [
    "apply_style",
    "palette",
    "plot_cost_comparison",
    "plot_latency_vs_concurrency",
    "plot_quality_retention",
    "plot_throughput_vs_concurrency",
]
