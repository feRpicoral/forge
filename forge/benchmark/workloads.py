"""Prompt-length statistics for documenting workload shape.

The benchmark harness offloads dataset download and prompt sampling to vLLM's
``bench serve`` command — it handles ShareGPT, Sonnet, random, and a few others
natively. This module exists for a separate, narrower purpose: after a run, we
read back the recorded input/output lengths and surface summary statistics in
the methodology section of the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median, stdev


@dataclass(frozen=True)
class LengthSummary:
    count: int
    mean: float
    median: float
    p99: float
    stdev: float
    min_value: int
    max_value: int

    @classmethod
    def from_values(cls, values: list[int]) -> LengthSummary | None:
        if not values:
            return None
        sorted_values = sorted(values)
        p99_idx = max(0, round(0.99 * (len(sorted_values) - 1)))
        return cls(
            count=len(sorted_values),
            mean=float(mean(sorted_values)),
            median=float(median(sorted_values)),
            p99=float(sorted_values[p99_idx]),
            stdev=float(stdev(sorted_values)) if len(sorted_values) > 1 else 0.0,
            min_value=sorted_values[0],
            max_value=sorted_values[-1],
        )
