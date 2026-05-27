"""Launch the vLLM OpenAI-compatible server with the resolved env config.

Usage:
    python -m scripts.serve         # prints + execs vllm serve
    python -m scripts.serve --dry-run  # prints the command, does not exec

The script ``execvp``s into ``vllm`` so signal handling (Ctrl-C, SIGTERM from
Docker / RunPod) maps directly onto the vLLM process — no Python wrapper to
intercept and mis-handle.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from forge.serving.config import ServingConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and exit without launching vllm.",
    )
    args = parser.parse_args(argv)

    config = ServingConfig.from_env()
    command = config.to_vllm_args()

    _emit_summary(config, command)

    if args.dry_run:
        return 0

    if config.hf_home:
        Path(config.hf_home).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = config.hf_home

    os.execvp(command[0], command)


def _emit_summary(config: ServingConfig, command: list[str]) -> None:
    print("[forge] serving config", file=sys.stderr)
    print(f"  model_id              = {config.model_id}", file=sys.stderr)
    print(f"  quantization          = {config.quantization}", file=sys.stderr)
    print(f"  max_model_len         = {config.max_model_len}", file=sys.stderr)
    print(f"  max_num_seqs          = {config.max_num_seqs}", file=sys.stderr)
    print(f"  host:port             = {config.host}:{config.port}", file=sys.stderr)
    print(f"  gpu_memory_utilization= {config.gpu_memory_utilization}", file=sys.stderr)
    print(f"  kv_cache_dtype        = {config.kv_cache_dtype}", file=sys.stderr)
    print(f"  tensor_parallel_size  = {config.tensor_parallel_size}", file=sys.stderr)
    print(f"  base_url              = {config.base_url}", file=sys.stderr)
    print("", file=sys.stderr)
    print(f"[forge] exec: {' '.join(command)}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
