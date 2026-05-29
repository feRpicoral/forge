"""Generate illustrative bench + eval JSONs so the chart pipeline
can be exercised end-to-end before the paid run lands real data.

Numbers are plausible (in the right shape and order of magnitude for Llama
3.1 8B on RTX A5000) but **not measured**. They are clearly marked illustrative
and are replaced wholesale by the real RunPod outputs.

Run once:
    uv run python -m scripts._gen_illustrative
"""

from __future__ import annotations

import json
from pathlib import Path

# Concurrency levels, must match configs/bench-full*.yaml.
CONCURRENCY_LEVELS = [1, 4, 16, 32, 64]

# Plausible throughput curve for Llama 3.1 8B on RTX A5000, BF16 vs AWQ-INT4.
# AWQ wins from c≥4 because INT4 cuts memory bandwidth pressure.
BF16 = {
    1: {
        "tput_total": 105.0,
        "tput_output": 50.0,
        "ttft_p50": 0.050,
        "ttft_p99": 0.080,
        "tpot_p50": 0.020,
        "tpot_p99": 0.028,
    },
    4: {
        "tput_total": 360.0,
        "tput_output": 175.0,
        "ttft_p50": 0.075,
        "ttft_p99": 0.130,
        "tpot_p50": 0.021,
        "tpot_p99": 0.031,
    },
    16: {
        "tput_total": 1380.0,
        "tput_output": 700.0,
        "ttft_p50": 0.210,
        "ttft_p99": 0.420,
        "tpot_p50": 0.023,
        "tpot_p99": 0.038,
    },
    32: {
        "tput_total": 2700.0,
        "tput_output": 1380.0,
        "ttft_p50": 0.420,
        "ttft_p99": 0.880,
        "tpot_p50": 0.025,
        "tpot_p99": 0.045,
    },
    64: {
        "tput_total": 4750.0,
        "tput_output": 2400.0,
        "ttft_p50": 0.820,
        "ttft_p99": 1.450,
        "tpot_p50": 0.027,
        "tpot_p99": 0.054,
    },
}

AWQ = {
    1: {
        "tput_total": 165.0,
        "tput_output": 82.0,
        "ttft_p50": 0.045,
        "ttft_p99": 0.072,
        "tpot_p50": 0.012,
        "tpot_p99": 0.018,
    },
    4: {
        "tput_total": 615.0,
        "tput_output": 300.0,
        "ttft_p50": 0.060,
        "ttft_p99": 0.110,
        "tpot_p50": 0.013,
        "tpot_p99": 0.020,
    },
    16: {
        "tput_total": 2350.0,
        "tput_output": 1180.0,
        "ttft_p50": 0.150,
        "ttft_p99": 0.320,
        "tpot_p50": 0.014,
        "tpot_p99": 0.024,
    },
    32: {
        "tput_total": 5450.0,
        "tput_output": 2800.0,
        "ttft_p50": 0.280,
        "ttft_p99": 0.620,
        "tpot_p50": 0.015,
        "tpot_p99": 0.028,
    },
    64: {
        "tput_total": 10300.0,
        "tput_output": 5200.0,
        "ttft_p50": 0.500,
        "ttft_p99": 0.910,
        "tpot_p50": 0.016,
        "tpot_p99": 0.034,
    },
}


def bench_payload(
    model: str, variant: str, concurrency: int, params: dict[str, float]
) -> dict[str, object]:
    """vLLM bench-serve-compatible payload."""
    return {
        "_illustrative": True,
        "_note": "Plausible-but-NOT-measured. Replaced by real RunPod data.",
        "date": "illustrative",
        "endpoint_type": "openai-chat",
        "backend": "openai-chat",
        "label": None,
        "model_id": model,
        "tokenizer_id": model,
        "num_prompts": 256,
        "request_rate": float("inf"),
        "burstiness": 1.0,
        "max_concurrency": concurrency,
        "duration": 256.0 / max(params["tput_output"] / 100.0, 0.1),
        "completed": 256,
        "total_input_tokens": 256 * 200,
        "total_output_tokens": 256 * 80,
        "request_throughput": params["tput_output"] / 80.0,  # rough rate
        "request_goodput": None,
        "output_throughput": params["tput_output"],
        "total_token_throughput": params["tput_total"],
        "max_output_tokens_per_s": params["tput_output"] * 1.15,
        "max_concurrent_requests": concurrency * 1.05,
        "mean_ttft_ms": params["ttft_p50"] * 1000 * 1.1,
        "median_ttft_ms": params["ttft_p50"] * 1000,
        "std_ttft_ms": params["ttft_p50"] * 1000 * 0.25,
        "p99_ttft_ms": params["ttft_p99"] * 1000,
        "mean_tpot_ms": params["tpot_p50"] * 1000 * 1.05,
        "median_tpot_ms": params["tpot_p50"] * 1000,
        "std_tpot_ms": params["tpot_p50"] * 1000 * 0.2,
        "p99_tpot_ms": params["tpot_p99"] * 1000,
        "mean_itl_ms": params["tpot_p50"] * 1000 * 0.95,
        "median_itl_ms": params["tpot_p50"] * 1000 * 0.9,
        "std_itl_ms": params["tpot_p50"] * 1000 * 0.18,
        "p99_itl_ms": params["tpot_p99"] * 1000 * 0.9,
    }


def write_bench(model: str, variant: str, base_dir: Path) -> None:
    name = f"full-{variant}"
    out = base_dir / variant
    out.mkdir(parents=True, exist_ok=True)
    curve = BF16 if variant == "bf16" else AWQ
    for c, params in curve.items():
        payload = bench_payload(model, variant, c, params)
        path = out / f"{name}-c{c:04d}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_eval(base_dir: Path) -> None:
    # Plausible Llama 3.1 8B Instruct scores; AWQ retains ~98%.
    base_dir.mkdir(parents=True, exist_ok=True)
    bf16 = {
        "_illustrative": True,
        "results": {
            "hellaswag": {"acc,none": 0.5872, "acc_norm,none": 0.7891},
            "mmlu": {"acc,none": 0.6810},
            "gsm8k": {"exact_match,strict-match": 0.8194, "exact_match,flexible-extract": 0.8237},
        },
        "n-shot": {"hellaswag": 5, "mmlu": 5, "gsm8k": 5},
        "config": {
            "model": "local-completions",
            "model_args": "model=meta-llama/Llama-3.1-8B-Instruct",
        },
    }
    awq = {
        "_illustrative": True,
        "results": {
            "hellaswag": {"acc,none": 0.5743, "acc_norm,none": 0.7782},
            "mmlu": {"acc,none": 0.6689},
            "gsm8k": {"exact_match,strict-match": 0.7999, "exact_match,flexible-extract": 0.8051},
        },
        "n-shot": {"hellaswag": 5, "mmlu": 5, "gsm8k": 5},
        "config": {
            "model": "local-completions",
            "model_args": "model=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        },
    }
    (base_dir / "full-bf16.json").write_text(json.dumps(bf16, indent=2), encoding="utf-8")
    (base_dir / "full-awq.json").write_text(json.dumps(awq, indent=2), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bench_root = repo_root / "results" / "bench" / "illustrative"
    eval_root = repo_root / "results" / "eval" / "illustrative"

    write_bench("meta-llama/Llama-3.1-8B-Instruct", "bf16", bench_root)
    write_bench("hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", "awq", bench_root)
    write_eval(eval_root)
    print(f"wrote illustrative bench to {bench_root}")
    print(f"wrote illustrative eval to {eval_root}")


if __name__ == "__main__":
    main()
