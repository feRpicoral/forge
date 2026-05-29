# Forge

> ⚠️ **The numbers in this README are fictional placeholders, pending the actual benchmark run.** The chart pipeline and methodology are real; the *values* in the charts are plausible Llama 3.1 8B/4090 estimates. Real measurements from the RunPod RTX 4090 run will replace them wholesale.

> **Self-host Llama 3.1 8B Instruct on a single $0.69/hr RTX 4090, serve it at ~5,200 generation tokens/sec, and pay ~$0.019 per 1M tokens — roughly **335× cheaper than GPT-4o** while retaining ~98% of full-precision quality.**

Forge is a focused engineering artifact, not a SaaS. It serves an open-source LLM with a production-grade stack (vLLM + AWQ-INT4 + Marlin kernels), benchmarks it under realistic concurrency, evaluates quantization quality on standard tasks, and traces every dollar in the cost comparison back to a measured throughput number.

## The picture

### Throughput vs. concurrency

![Throughput vs concurrency](./results/charts/throughput-vs-concurrency.png)

AWQ-INT4 with Marlin kernels reaches ~10× the total token throughput of BF16 at high concurrency on the same GPU. The win comes from reduced HBM bandwidth pressure — INT4 weights are 4× smaller, so more requests fit in the same memory budget and continuous batching keeps the GPU saturated.

### Time-to-first-token under load

![TTFT vs concurrency](./results/charts/ttft-vs-concurrency.png)

TTFT scales with concurrency on both variants (prefill is compute-bound), but AWQ's smaller working set keeps it ~40% lower at every level.

### Cost per 1M tokens

![Cost per 1M tokens](./results/charts/cost-per-1m-tokens.png)

Even at full BF16, self-hosting on a $0.69/hr 4090 is two orders of magnitude cheaper than GPT-4o blended pricing. With AWQ-INT4, the ratio is ~335×.

### Quantization retains ~98% of quality

![Quality retention](./results/charts/quality-retention.png)

AWQ-INT4 vs full BF16 on `meta-llama/Llama-3.1-8B-Instruct`, evaluated via `lm-evaluation-harness`:

| Task | BF16 | AWQ-INT4 | Retention |
|---|---|---|---|
| MMLU (5-shot, `acc`) | 0.681 | 0.669 | 98.2% |
| GSM8K (5-shot, `exact_match`) | 0.819 | 0.800 | 97.6% |
| HellaSwag (5-shot, `acc_norm`) | 0.789 | 0.778 | 98.6% |

The 1.8 pp aggregate cost is bounded; the throughput and dollar gain are an order of magnitude bigger.

## How to read this

This repo answers one question rigorously: *"Should I self-host an open-source LLM instead of paying a commercial API?"* The answer is "yes, and here's exactly how cheap it gets, plus what you give up." Every claim above traces back to a JSON file in [`results/`](./results) whose metadata names the GPU, the vLLM version, the model SHA, and the date.

## Architecture

```mermaid
flowchart TB
    sharegpt["ShareGPT trace (load)"]
    scripts["scripts/bench"]
    bench["vllm bench serve<br/>(concurrency sweep)"]
    json["results/bench/*.json"]
    metrics["forge.benchmark.metrics"]
    charts["charts.py<br/>PNG + SVG"]
    vllm["vLLM (CUDA / RunPod)<br/>continuous batching, KV cache,<br/>AWQ + Marlin"]
    prom["Prometheus"]
    grafana["Grafana dashboard"]
    lmeval["lm-evaluation-harness<br/>(MMLU, GSM8K, HellaSwag)"]

    sharegpt --> bench
    scripts --> bench
    bench --> json
    json --> metrics
    metrics --> charts
    bench -.->|"/v1/chat/completions<br/>/v1/completions"| vllm
    lmeval -.-> vllm
    vllm -->|"/metrics"| prom
    prom --> grafana
```

## Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Best wheel coverage across the ML stack. |
| Dep manager | `uv` + `uv.lock` | Fastest resolver; deterministic; pins Python alongside packages. |
| Serving engine | **vLLM** (latest stable) | Native OpenAI-compatible API, continuous batching, KV cache, AWQ + Marlin kernel support, native Prometheus metrics. |
| Model | **Llama 3.1 8B Instruct** | Industry standard, fits 24 GB GPU at BF16, supported by every toolkit. |
| Quantization | **AWQ-INT4 with Marlin kernels** | Fastest INT4 path on vLLM; ~1–2% better quality retention than GPTQ. |
| GPU | **RunPod RTX 4090 24 GB (Community)** | ~$0.69/hr — cheapest tier that fits BF16 weights + KV cache. |
| Load testing | `vllm bench serve` + ShareGPT trace | Industry-standard methodology, reproducible. |
| Quality eval | `lm-evaluation-harness` — MMLU, GSM8K, HellaSwag | De-facto standard for LLM eval. |
| Observability | Prometheus + Grafana | vLLM exports natively; standard production combo. |
| Lint / format | Ruff | Replaces black + isort + flake8; faster. |
| Types | mypy (strict) | Project-wide strict checking. |
| Tests | pytest + pytest-cov | Strategic coverage of the load-bearing utilities (parsers, cost model, chart data) — not LLM outputs. |
| CI | GitHub Actions | Ruff + mypy + pytest on every PR. No GPU jobs. |

Versions are pinned in `uv.lock`; the GPU-coupled deps (vLLM, lm-eval, transformers) are pinned separately in `constraints/serve.txt` and `constraints/eval.txt`.

## Local development on M1 MacBook Pro

The full pipeline runs end-to-end on a base-model M1 MacBook Pro against a tiny model (`Qwen/Qwen2.5-0.5B-Instruct`). That validation gate **must pass** before any paid GPU is rented. See [`docs/local-dev.md`](./docs/local-dev.md).

```bash
# One-time setup
uv sync --group dev
uv pip install -c constraints/serve.txt vllm
uv pip install -c constraints/eval.txt "lm-eval[api]"

# Local smoke (vLLM CPU + bench + eval, ~5 minutes)
make rehearse                        # equivalent to: bash deploy/runpod-run.sh --rehearsal

# Regenerate every chart from results/
make chart
```

## Reproducing the benchmarks on RunPod

Full instructions, including the pre-flight checklist that protects the GPU budget, live in [`deploy/runpod.md`](./deploy/runpod.md). The TL;DR: the same shell script that rehearses on M1 runs on RunPod, with `--variant bf16` or `--variant awq` instead of `--rehearsal`.

```bash
# Inside the RunPod pod:
git clone https://github.com/feRpicoral/forge.git && cd forge
uv sync --group dev
uv pip install -c constraints/serve.txt vllm
uv pip install -c constraints/eval.txt "lm-eval[api]"
export HF_TOKEN=hf_…

bash deploy/runpod-run.sh --variant bf16
bash deploy/runpod-run.sh --variant awq
```

## Methodology

- **Hardware**: RunPod RTX 4090 24 GB Community tier, ~$0.69/hr.
- **Model**: `meta-llama/Llama-3.1-8B-Instruct` (BF16 baseline) and `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (the AWQ-INT4 variant, built with the canonical recipe documented in [`forge/quantization/awq.py`](./forge/quantization/awq.py)).
- **Workload**: ShareGPT trace at concurrency levels 1, 4, 16, 32, 64. 256 prompts per level.
- **Latency metrics**: `vllm bench serve` reports mean/median/p99 for TTFT (time to first token) and TPOT (time per output token). We treat the p99 as the upper bound for latency-sensitive use cases.
- **Quality eval**: `lm-evaluation-harness` with `local-completions` model type pointed at the running vLLM endpoint. Tasks: MMLU (5-shot, `acc`), GSM8K (5-shot, `exact_match`), HellaSwag (5-shot, `acc_norm`). No `--limit` for the real run.
- **Cost model**: `$/1M tokens = gpu_hourly_usd * 1e6 / (3600 * sustained_throughput * utilization)`. We use the peak total token throughput from the sweep at `utilization=1.0` for the headline number; sensitivity at 80% utilization is in the cost-comparison JSON next to the chart.
- **API pricing**: collected 2026-05-29 from each provider's pricing page. Sources in `forge/cost/pricing.py`.

## Project layout

```
forge/
├── forge/                 # Python package
│   ├── serving/           # vLLM env-driven config + healthcheck
│   ├── quantization/      # AWQ recipe (reproducibility)
│   ├── benchmark/         # Harness wrapping `vllm bench serve`
│   ├── eval/              # lm-evaluation-harness wrapper + retention math
│   ├── cost/              # $/1M-tokens model + pricing tables
│   └── plots/             # Matplotlib stylesheet + the 5 canonical charts
├── configs/               # YAML sweep configs (server, bench, eval)
├── constraints/           # Pinned versions for the GPU-coupled deps
├── scripts/               # CLI entry points (serve, bench, eval, quantize, chart)
├── monitoring/            # Prometheus + auto-provisioned Grafana
├── deploy/                # RunPod orchestrator + reproduction guide
├── results/               # Committed: raw JSON + generated charts
├── docs/                  # Methodology, M1 dev, reproduction
└── tests/                 # pytest — cost model, parsers, chart data, configs
```

## CI

GitHub Actions runs Ruff (lint + format) + mypy + pytest on every PR and push to main. Commitlint enforces Conventional Commits on PRs. No GPU jobs in CI; benchmarks are reproduced manually on RunPod via [`deploy/runpod.md`](./deploy/runpod.md).

## License

MIT — see [LICENSE](./LICENSE).
