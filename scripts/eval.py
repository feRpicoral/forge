"""CLI for running an lm-evaluation-harness suite.

Usage:
    python -m scripts.eval --config configs/eval-smoke.yaml
    python -m scripts.eval --config configs/eval-smoke.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forge.eval.config import load_suite
from forge.eval.runner import run_suite, validate_output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run and args.verify_only:
        parser.error("--dry-run and --verify-only cannot be combined")

    suite = load_suite(args.config)

    print(f"[forge] eval suite       = {suite.name}", file=sys.stderr)
    print(f"[forge] description      = {suite.description}", file=sys.stderr)
    print(f"[forge] variant          = {suite.variant}", file=sys.stderr)
    print(f"[forge] model            = {suite.model}", file=sys.stderr)
    print(f"[forge] base_url         = {suite.base_url}", file=sys.stderr)
    print(f"[forge] tasks            = {suite.tasks}", file=sys.stderr)
    print(f"[forge] num_fewshot      = {suite.num_fewshot}", file=sys.stderr)
    print(f"[forge] batch_size       = {suite.batch_size}", file=sys.stderr)
    print(f"[forge] limit            = {suite.limit}", file=sys.stderr)
    print(f"[forge] result_path      = {suite.result_path}", file=sys.stderr)
    print("", file=sys.stderr)

    if args.dry_run:
        print("[forge] dry-run, exiting without launching lm-eval.", file=sys.stderr)
        return 0
    if args.verify_only:
        validate_output(suite)
        print("[forge] result JSON validation OK.", file=sys.stderr)
        return 0

    outcome = run_suite(suite)
    status = "OK" if outcome.succeeded else f"FAIL(rc={outcome.returncode})"
    print(f"[forge] eval {status} → {outcome.result_path}", file=sys.stderr)
    return 0 if outcome.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
