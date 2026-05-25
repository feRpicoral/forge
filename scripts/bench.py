"""CLI entry point for running a benchmark sweep.

Usage:
    python -m scripts.bench --config configs/bench-smoke.yaml
    python -m scripts.bench --config configs/bench-smoke.yaml --dry-run

The vLLM server at ``base_url`` must already be running and serving ``model``.
Start it in another shell with ``make serve``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forge.benchmark.config import load_sweep
from forge.benchmark.runner import run_sweep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a benchmark sweep YAML (e.g. configs/bench-smoke.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the config and print the plan without invoking vllm.",
    )
    args = parser.parse_args(argv)

    sweep = load_sweep(args.config)

    print(f"[forge] sweep              = {sweep.name}", file=sys.stderr)
    print(f"[forge] description        = {sweep.description}", file=sys.stderr)
    print(f"[forge] model              = {sweep.model}", file=sys.stderr)
    print(f"[forge] base_url           = {sweep.base_url}", file=sys.stderr)
    print(f"[forge] dataset            = {sweep.dataset.name}", file=sys.stderr)
    print(f"[forge] num_prompts        = {sweep.dataset.num_prompts}", file=sys.stderr)
    print(f"[forge] concurrency_levels = {sweep.concurrency_levels}", file=sys.stderr)
    print(f"[forge] result_dir         = {sweep.result_dir}", file=sys.stderr)
    print("", file=sys.stderr)

    if args.dry_run:
        print("[forge] dry-run, exiting without launching vllm.", file=sys.stderr)
        return 0

    outcomes = run_sweep(sweep)

    print("", file=sys.stderr)
    print("[forge] sweep summary", file=sys.stderr)
    for outcome in outcomes:
        status = "OK" if outcome.succeeded else f"FAIL(rc={outcome.returncode})"
        print(
            f"  c={outcome.concurrency:>4d}  {status:>10s}  {outcome.result_path}", file=sys.stderr
        )

    failures = [o for o in outcomes if not o.succeeded]
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
