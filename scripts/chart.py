"""Regenerate every Forge chart from the committed ``results/`` artifacts.

Usage:
    python -m scripts.chart                       # rebuild from results/ defaults
    python -m scripts.chart --output results/charts

When no real benchmark data is committed yet, this script emits illustrative
charts from synthetic data shipped under ``results/illustrative/`` so the
chart pipeline itself can be validated. Real charts replace those files after
the paid RunPod run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.benchmark.metrics import BenchmarkRow, load_results
from forge.cost.model import build_self_hosted, compare
from forge.eval.compare import compute_deltas, parse_lm_eval_output
from forge.plots.charts import (
    plot_cost_comparison,
    plot_latency_vs_concurrency,
    plot_quality_retention,
    plot_throughput_vs_concurrency,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory of benchmark + eval results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/charts"),
        help="Directory the regenerated charts are written into.",
    )
    args = parser.parse_args(argv)

    bench_root = args.results_root / "bench"
    eval_root = args.results_root / "eval"

    bf16_dir = _first_existing([bench_root / "full-bf16", bench_root / "illustrative" / "bf16"])
    awq_dir = _first_existing([bench_root / "full-awq", bench_root / "illustrative" / "awq"])
    eval_dir = _first_existing([eval_root / "full", eval_root / "illustrative"])

    if not bf16_dir or not awq_dir:
        print(
            "[forge] no bench results found. Run the paid sweep or use illustrative data.",
            file=sys.stderr,
        )
        return 1

    print(f"[forge] bf16  dir = {bf16_dir}", file=sys.stderr)
    print(f"[forge] awq   dir = {awq_dir}", file=sys.stderr)
    print(f"[forge] eval  dir = {eval_dir}", file=sys.stderr)
    print(f"[forge] output    = {args.output}", file=sys.stderr)

    bf16_rows = _load_bench_dir(bf16_dir)
    awq_rows = _load_bench_dir(awq_dir)

    series: dict[str, list[BenchmarkRow]] = {"bf16": bf16_rows, "awq": awq_rows}

    paths = []
    paths.append(plot_throughput_vs_concurrency(series=series, output_dir=args.output))
    paths.append(plot_latency_vs_concurrency(series=series, metric="ttft", output_dir=args.output))
    paths.append(plot_latency_vs_concurrency(series=series, metric="tpot", output_dir=args.output))

    # Cost comparison uses the headline throughput at the chosen concurrency.
    awq_peak = max(awq_rows, key=lambda r: r.total_token_throughput)
    bf16_peak = max(bf16_rows, key=lambda r: r.total_token_throughput)
    cost = compare(
        [
            build_self_hosted(
                label="Forge — BF16 on 4090",
                gpu_tier_key="runpod-rtx-4090-community",
                sustained_throughput_tps=bf16_peak.total_token_throughput,
            ),
            build_self_hosted(
                label="Forge — AWQ-INT4 on 4090",
                gpu_tier_key="runpod-rtx-4090-community",
                sustained_throughput_tps=awq_peak.total_token_throughput,
            ),
        ],
        ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-6", "claude-haiku-4-5"],
        input_share=0.5,
    )
    paths.append(plot_cost_comparison(comparison=cost, output_dir=args.output))

    # Also write the cost-comparison JSON next to the chart for traceability.
    (args.output / "cost-per-1m-tokens.json").write_text(
        json.dumps(cost.to_dict(), indent=2), encoding="utf-8"
    )

    if eval_dir:
        bf16_eval = eval_dir / "full-bf16.json"
        awq_eval = eval_dir / "full-awq.json"
        if bf16_eval.exists() and awq_eval.exists():
            baseline = parse_lm_eval_output(bf16_eval)
            candidate = parse_lm_eval_output(awq_eval)
            delta = compute_deltas(baseline, candidate)
            paths.append(plot_quality_retention(delta=delta, output_dir=args.output))
            (args.output / "quality-retention.json").write_text(
                json.dumps(delta.to_dict(), indent=2), encoding="utf-8"
            )
        else:
            print(
                f"[forge] eval files missing in {eval_dir}; skipping quality chart.",
                file=sys.stderr,
            )

    print("", file=sys.stderr)
    print("[forge] charts written:", file=sys.stderr)
    for p in paths:
        print(f"  {p.png}", file=sys.stderr)
        print(f"  {p.svg}", file=sys.stderr)

    return 0


def _first_existing(candidates: list[Path]) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_bench_dir(directory: Path) -> list[BenchmarkRow]:
    """Load every smoke/full JSON in a directory, ordered by concurrency.

    Filename format: ``{name}-c{NNNN}.json``. We derive concurrency from the
    filename, not by re-parsing the JSON.
    """
    rows: list[BenchmarkRow] = []
    name = None
    concurrencies: list[int] = []
    for path in sorted(directory.glob("*-c[0-9][0-9][0-9][0-9].json")):
        stem = path.stem
        prefix, _, c_part = stem.rpartition("-c")
        try:
            concurrency = int(c_part)
        except ValueError:
            continue
        name = prefix
        concurrencies.append(concurrency)
    if name is None or not concurrencies:
        return rows
    return load_results(directory, sorted(set(concurrencies)), name=name)


if __name__ == "__main__":
    raise SystemExit(main())
