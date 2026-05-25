# Forge

> Production-grade LLM inference serving & optimization — benchmarks, quantization, cost analysis.

**Status:** scaffolding. Impact lead, benchmarks, and charts will land at Phase 9 of the [implementation plan](./DECISIONS.md).

This is a portfolio piece demonstrating that an open-source LLM can be served and optimized in production — not a SaaS product. The deliverable is a reproducible benchmark suite and a defensible cost vs. quality study comparing self-hosted Llama 3.1 8B (AWQ-INT4 on vLLM) against commercial APIs.

## What's here

- **Serving layer**: vLLM with continuous batching, KV cache, and OpenAI-compatible streaming API.
- **Quantization**: AWQ-INT4 with Marlin kernels (vLLM-native).
- **Benchmark harness**: throughput, TTFT (p50/p95), TPOT (p50/p95) under realistic concurrency.
- **Quality eval**: `lm-evaluation-harness` (MMLU, GSM8K, HellaSwag) — full precision vs quantized delta.
- **Cost model**: $/1M tokens self-hosted vs GPT-4o / Claude.
- **Observability**: Prometheus metrics + Grafana dashboard.

## Stack

Python 3.12, vLLM, AutoAWQ, lm-evaluation-harness, Prometheus, Grafana, Docker. Linted with Ruff, type-checked with mypy, tested with pytest.

## Local development (M1 MacBook Pro)

This repo is developed and rehearsed on a base-model M1 MacBook Pro. The full pipeline runs locally against a tiny model (`Qwen/Qwen2.5-0.5B-Instruct`) for smoke testing before any paid GPU rental. See [`docs/local-dev.md`](./docs/local-dev.md) for the vLLM CPU build instructions and the Ollama fallback path.

```bash
# Install Python 3.12 and sync deps
uv sync

# Run linters and tests
make lint
make test

# (Phase 1+) Launch vLLM CPU against the tiny model
make serve

# (Phase 2+) Smoke-run the benchmark harness
make bench CONFIG=bench-smoke
```

## Reproducing the benchmarks

The full benchmark methodology, hardware, model SHAs, and exact vLLM version live in [`docs/methodology.md`](./docs/methodology.md). To reproduce on a rented GPU, see [`deploy/runpod.md`](./deploy/runpod.md).

## Project layout

```
forge/
├── forge/                 # Python package
│   ├── serving/           # vLLM server config + healthcheck
│   ├── quantization/      # AWQ recipe
│   ├── benchmark/         # Harness wrapping vLLM's benchmark_serving
│   ├── eval/              # lm-evaluation-harness wrapper + comparison
│   ├── cost/              # $/1M-tokens model + pricing tables
│   └── plots/             # Chart generation (matplotlib)
├── configs/               # Server + benchmark YAML configs
├── scripts/               # CLI entry points (serve, bench, eval, quantize, chart)
├── tests/                 # pytest — cost model, parsers, metrics
├── monitoring/            # Prometheus + Grafana
├── deploy/                # RunPod scripts + HF Spaces app
├── results/               # Committed benchmark JSON + generated charts
└── docs/                  # Methodology, local dev, reproduction
```

## CI

GitHub Actions runs Ruff (lint + format), mypy, and pytest on every PR and push to main. No GPU jobs in CI — benchmarks are reproduced on rented GPUs by following the methodology docs.

## License

MIT — see [LICENSE](./LICENSE).
