"""CLI entry point for AWQ quantization.

Usage:
    python -m scripts.quantize --source meta-llama/Llama-3.1-8B-Instruct \\
                               --output ./out/llama-3.1-8b-awq-int4

    python -m scripts.quantize --dry-run   # prints the resolved config and exits

Requires a CUDA GPU + ``autoawq`` installed. On M1, use the pre-quantized
community checkpoint instead:
    hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forge.quantization.awq import AWQConfig


def build_config(argv: list[str] | None = None) -> tuple[AWQConfig, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model identifier of the source full-precision model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./out/llama-3.1-8b-awq-int4"),
        help="Local directory the quantized model is written to.",
    )
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=128)
    parser.add_argument(
        "--no-zero-point",
        action="store_true",
        help="Disable asymmetric quantization (default: enabled).",
    )
    parser.add_argument(
        "--version",
        choices=["GEMM", "GEMV"],
        default="GEMM",
        help="AWQ kernel family. GEMM is faster on modern GPUs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config and exit without quantizing.",
    )

    args = parser.parse_args(argv)
    config = AWQConfig(
        source_model=args.source,
        output_path=args.output,
        w_bit=args.w_bit,
        q_group_size=args.q_group_size,
        zero_point=not args.no_zero_point,
        version=args.version,
    )
    return config, args.dry_run


def main(argv: list[str] | None = None) -> int:
    config, dry_run = build_config(argv)

    print("[forge] quantize", file=sys.stderr)
    print(config.describe(), file=sys.stderr)
    print("", file=sys.stderr)

    if dry_run:
        print("[forge] dry-run, exiting without quantizing.", file=sys.stderr)
        return 0

    from forge.quantization.awq import quantize

    output_path = quantize(config)
    print(f"[forge] quantized model written to: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
