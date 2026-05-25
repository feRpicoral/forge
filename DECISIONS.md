# DECISIONS.md

> Forward-looking record of technical choices for Forge, with the alternatives considered and the reasoning. Updated incrementally as the project progresses. Each decision pins the *exact* version actually used after `uv sync` completes — never a speculative range.

## Scope

Forge is a portfolio piece, not a SaaS. It serves an open-source LLM via a production-grade stack and produces a defensible benchmark + cost analysis. Decisions favor: (1) industry-standard tooling that signals familiarity with how serving is done in production, (2) reproducibility, (3) cost-conscious paid GPU usage.

## Stack overview

| Layer | Choice | Status |
|---|---|---|
| Language | Python 3.12 | locked |
| Dep manager | uv + `uv.lock` | locked |
| Serving engine | vLLM | locked |
| Model | Llama 3.1 8B Instruct (paid run) / Qwen 2.5 0.5B Instruct (M1 smoke) | locked |
| Quantization | AWQ-INT4 with Marlin kernels | locked |
| GPU (benchmarks) | RunPod RTX 4090 24 GB Community | locked |
| Load testing | vLLM `benchmark_serving.py` + ShareGPT trace | locked |
| Quality eval | `lm-evaluation-harness` — MMLU + GSM8K + HellaSwag | locked |
| Observability | Prometheus + Grafana | locked |
| Lint / format | Ruff | locked |
| Type checker | mypy (strict) | locked |
| Tests | pytest + pytest-cov | locked |
| CI | GitHub Actions | locked |
| HF demo | Gradio on Spaces ZeroGPU | planned |

Exact versions are recorded under "Pinned versions" after the first successful `uv sync`.

## Decisions

### Python 3.12
- **Why**: vLLM and the broader ML ecosystem ship reliable wheels for 3.12. 3.13 still has rough edges in transformers + torch.
- **Alternatives**: 3.11 (older, less reason to choose), 3.13 (too new for the stack).

### `uv` over `poetry` / `pip-tools`
- **Why**: Recommended by the project brief. Fastest resolver, single CLI, deterministic `uv.lock`, native Python version management.
- **Alternatives**: poetry (slower, heavier), pip-tools (more manual), rye (similar to uv but smaller community).

### Serving engine: vLLM
- **Why**: Best ecosystem and DX for portfolio work. Native OpenAI-compatible API. First-class AWQ + GPTQ + FP8 support. Native Prometheus metrics. Industry-standard `benchmark_serving.py` is the methodology most readers will recognize.
- **Alternatives considered**:
  - **SGLang**: ~29% throughput edge over vLLM on prefix-shared workloads (RAG/agent/chat). Strong choice for RAG, but adds a separate deploy path that doesn't fit the scoped portfolio piece. Documented here, not benchmarked.
  - **TensorRT-LLM**: 8–13% higher throughput than vLLM at matched concurrency on H100. Requires per-model compilation (~28 min) and tight NVIDIA toolchain coupling. Higher friction, less suited to "model-update flexibility" narrative.
  - **TGI (Text Generation Inference)**: HuggingFace's serving. Mature but less SOTA on throughput than vLLM/SGLang in recent benchmarks.
- **Sources**:
  - [vLLM vs TensorRT-LLM vs SGLang: H100 Benchmarks (Spheron)](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/)
  - [SGLang vs vLLM (Particula)](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison)
  - [vLLM official docs](https://docs.vllm.ai/)

### Model: Llama 3.1 8B Instruct
- **Why**: Industry standard, fits comfortably on a single 24 GB GPU at BF16 (~16 GB weights + KV cache). Every quantization toolkit supports it. Gated on HF Hub — token validated during pre-flight.
- **Alternatives**: Mistral 7B Instruct (ungated fallback), Qwen 2.5 7B Instruct (ungated fallback). Either fallback selected if Llama gating becomes a blocker.

### Quantization: AWQ-INT4 (Marlin kernels)
- **Why**: AWQ + Marlin is the fastest INT4 path on vLLM and retains the most quality. Roughly 1–2% better benchmark scores than GPTQ at equivalent compression. Marlin gives ~10x speedup over baseline INT4 kernels.
- **Alternatives considered**:
  - **GPTQ-INT4**: slightly lower quality, slower kernels in vLLM.
  - **FP8 (weight + activation)**: nearly indistinguishable from BF16 quality on most tasks; ~2x latency speedup on H100/L40S. Requires Hopper or Ada-Lovelace native FP8 hardware. **RTX 4090 has FP8 compute but not FP8 KV cache** — and is out of budget anyway. Documented as a future direction.
  - **bitsandbytes NF4**: simpler, slower, lower quality retention.
  - **GGUF**: optimized for llama.cpp/CPU edge. Not the right target for vLLM serving.
- **Sources**:
  - [LLM Quantization Explained (VRLA Tech)](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/)
  - [vLLM Quantization Guide (Jarvis Labs)](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks)

### GPU: RunPod RTX 4090 24 GB Community
- **Why**: Cheapest tier at ~$0.34/hr that fits Llama 3.1 8B BF16 (16 GB weights + KV cache headroom). Budget-aligned with the user's tight cost cap. Community tier accepted because the benchmark is short and re-runnable.
- **Alternatives**: A100 40 GB (~$1.19/hr — 3.5x cost), H100 80 GB (~$3+/hr — out of budget for the scoped run). Multi-tier sweep was rejected for budget reasons.

### Load testing: vLLM `benchmark_serving.py` + ShareGPT trace
- **Why**: Industry standard. Used in every vLLM throughput claim. Trace-driven (real prompt-length distributions) so results match reality. Reproducible — the script, the trace, and the seeds are all public.
- **Alternatives**: `llmperf` (Anyscale; similar but less aligned with vLLM-native metrics), `locust` (general, not LLM-aware), custom (reinventing the wheel, lower credibility).

### Quality eval: lm-evaluation-harness (MMLU + GSM8K + HellaSwag)
- **Why**: The de-facto standard for LLM eval. Cited in every paper. Subset is chosen to fit under ~45 min per model variant on a single 4090, keeping paid-GPU time bounded.
- **Alternatives**: Big-bench (too large for budget), HELM (heavier infra), custom task-specific eval (less defensible). A small hand-curated task-specific prompt set + LLM-as-judge is *added* on top of the standard subset for narrative color, not used as the primary claim.

### Observability: Prometheus + Grafana
- **Why**: vLLM exports Prometheus metrics natively. Grafana is the universal dashboard. Standard production combo — instantly recognizable on a portfolio.
- **Alternatives**: OpenTelemetry collector (more general; overkill here), custom dashboards (visually inconsistent, not industry-standard).

### Lint/format: Ruff (single tool)
- **Why**: Project brief specifies. Replaces black + isort + flake8. Faster, single config surface.
- **Alternatives**: black + isort + flake8 (legacy combo), pylint (slower, opinionated).

### Type checker: mypy strict
- **Why**: Project brief specifies. Strict mode catches the most real errors.
- **Alternatives**: pyright (faster, but mypy is the more conservative ecosystem choice).

### Tests: pytest + pytest-cov
- **Why**: Standard. Test surface is intentionally narrow — cost model, parsers, metrics aggregation, chart data shaping. Not testing LLM outputs themselves.
- **Coverage**: strategic, not 100%. CI enforces tests pass; no coverage threshold to avoid pressure to test trivial code.

### CI: GitHub Actions, single `ci.yml`
- **Why**: Matches the sibling projects' pattern (Sonar, Relay, Cite). One workflow runs Ruff + mypy + pytest on every PR and push. GPU-dependent steps (benchmarks, eval) are explicitly excluded — they are reproduced manually on RunPod following `docs/methodology.md`.

### HF Spaces demo: Gradio + ZeroGPU
- **Why**: Lowest-friction interactive demo. Gradio renders client-side, app code is small. ZeroGPU is free and time-slices an H100 per request — fine for a demo. Fallback: static replay of pre-recorded sessions.

## Pinned versions

To be filled in after the first successful `uv sync` on the M1.

```
python: <version>
ruff: <version>
mypy: <version>
pytest: <version>
pytest-cov: <version>
pre-commit: <version>
```

vLLM, transformers, torch, AutoAWQ, lm-evaluation-harness pins land in Phase 1+ as each module is added.

## Compatibility matrix (validated)

To be filled in after the first end-to-end M1 smoke and the first paid RunPod run. Critical pairs:
- torch ↔ CUDA
- torch ↔ vLLM
- vLLM ↔ transformers
- vLLM ↔ AutoAWQ
