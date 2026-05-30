"""Matplotlib stylesheet for Forge charts.

The goal is "engineering blog" aesthetics — clean axes, readable typography,
no chartjunk. Calling ``apply_style()`` once at the start of a script
configures matplotlib's rcParams; subsequent plots inherit them.

Colors are a color-blind-friendly palette adapted from the Okabe-Ito set,
with deliberate semantic mapping: blue = full precision, orange = quantized,
gray-tones = baselines and references.
"""

from __future__ import annotations

import matplotlib as mpl


def palette() -> dict[str, str]:
    """Named colors used throughout the charts.

    Returns a dict for explicit per-series color binding — avoid relying on
    cycler order so the BF16 series is always blue regardless of plot order.
    """
    return {
        "bf16": "#0072B2",  # Okabe-Ito blue
        "awq": "#D55E00",  # Okabe-Ito vermillion
        "fp8": "#009E73",  # Okabe-Ito green (held in reserve)
        "api_gpt4o": "#7570b3",
        "api_claude": "#1b9e77",
        "api_other": "#666666",
        "baseline": "#444444",
        "muted": "#888888",
        "ttft_p50": "#0072B2",
        "ttft_p95": "#56B4E9",
        "tpot_p50": "#D55E00",
        "tpot_p95": "#E69F00",
    }


def apply_style() -> None:
    """Install the Forge matplotlib defaults onto the global rcParams."""
    mpl.rcParams.update(
        {
            # Typography
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            # Axes & spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            # Lines & markers
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
            # Figure sizing & DPI
            "figure.figsize": (8.0, 5.0),
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.hashsalt": "forge",
            # Legend
            "legend.frameon": False,
            "legend.loc": "best",
        }
    )
