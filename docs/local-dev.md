# Local development on M1 MacBook Pro

This is the dev and rehearsal environment. The M1 cannot produce defensible benchmark numbers — no CUDA, no Hopper/Ada FP8, unified-memory bandwidth is not comparable to a 4090 — but it is fully capable of validating the entire pipeline end-to-end against a tiny model. The paid GPU run is exactly the configuration rehearsed locally.

## TL;DR

```bash
# One-time setup
uv sync --all-groups
pre-commit install --install-hooks --hook-type commit-msg

# Choose ONE backend for the local OpenAI-compatible server:

# A) Canonical: vLLM CPU build (slow, same engine as production)
docker compose up -d vllm-cpu       # builds + starts on first run; takes a while
# or: build vLLM CPU directly into your uv env — see "Building vLLM CPU on M1" below.

# B) Fast fallback: Ollama
brew install ollama
ollama serve &
ollama pull qwen2.5:0.5b
export OPENAI_BASE_URL=http://localhost:11434/v1   # point the harness at Ollama

# Validate
make check                          # ruff + mypy + pytest
make serve                          # (with backend A) launches vLLM via scripts/serve.py
python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').status)"
```

## Two layered backends

The benchmark harness, the eval pipeline, and the HF Spaces demo all talk to whatever is on the other end of an OpenAI-compatible base URL. That gives us two interchangeable backends on the M1:

### A) vLLM CPU — the canonical smoke

This is the *same engine* deployed on RunPod, just running on CPU with FP32/FP16. It exercises the exact code path (server entrypoint, OpenAI-compatible API, request scheduling, the bundled `benchmark_serving.py`). Slow — token throughput is a tiny fraction of a 4090 — but for 10–50 short prompts against a 125M–1.1B-parameter model it's fine, completing in minutes.

**Use it for:**

- Pre-flight smoke before any paid GPU run.
- Validating that `scripts/serve.py` builds the right argv for vLLM.
- Confirming the Prometheus `/metrics` endpoint shape matches what the Grafana dashboard expects.

**Tiny models that work well:**

- `Qwen/Qwen2.5-0.5B-Instruct` — ungated, modern tokenizer, our default for env defaults.
- `facebook/opt-125m` — tiniest reliable choice; OPT tokenizer is well-supported.
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — closer in shape to the production model but heavier.

**Do not** attempt to load Llama 3.1 8B BF16 on a base-model M1: 16 GB of unified memory is roughly the weight size alone with no room for activations or KV cache. That is the entire point of the paid run.

### B) Ollama — the fast fallback

Ollama runs natively on M1 via Metal, exposes an OpenAI-compatible API at `http://localhost:11434/v1`, and serves Q4-quantized 7B/8B models in 4–6 GB at 20+ tok/s on M1.

**Use it for:**

- Iterating on the HF Spaces Gradio demo (no need to wait for vLLM's CPU build between code changes).
- Exercising the harness against realistic streaming behavior without committing to a from-source vLLM build.
- Validating the chart pipeline against semi-realistic numbers when prototyping plots.

**It is a developer convenience, not the canonical smoke.** The benchmark methodology never reports Ollama numbers as if they were vLLM numbers.

## Installing vLLM on M1 (without Docker)

The PyPI wheel for vLLM works on darwin arm64 — no source build needed. Install it out-of-band into the uv-managed venv, against the pinned constraint file:

```bash
uv pip install -c constraints/serve.txt vllm
```

This pulls vLLM 0.11.0 plus a transformers 4.x pin. Without the constraint, uv would pull transformers 5.x, which crashes vLLM at startup with `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended` — see `DECISIONS.md` for the rationale.

vLLM is intentionally *not* a managed project dependency. Locking it inside `pyproject.toml` forces uv to resolve CUDA-only transitives (nvidia-cudnn-frontend etc.) for every platform in the lock, which fails on darwin. The constraint file pattern keeps the project lock clean and works on both macOS and Linux.

If the wheel install ever stops working (e.g. a future vLLM release drops macOS wheels), fall back to Ollama for harness validation:

```bash
brew install ollama
ollama serve &
ollama pull qwen2.5:0.5b
export OPENAI_BASE_URL=http://localhost:11434/v1
```

Upstream reference: <https://docs.vllm.ai/en/stable/getting_started/installation/cpu/>

## What the M1 can and cannot validate

| Concern | Validated on M1? |
|---|---|
| Harness end-to-end (request loop, streaming, JSON, parser) | yes |
| Quality eval pipeline (against tiny model — deltas are meaningless but wiring is exercised) | yes |
| Cost model (pure Python) | yes |
| Chart generation (matplotlib) | yes |
| `Dockerfile.cpu` (CPU build) | yes (built and started locally) |
| `Dockerfile` (CUDA, production) | builds only — cannot run locally |
| HF Spaces Gradio app | yes (pointed at Ollama) |
| CI (Ruff, mypy, pytest) | yes |
| Real throughput / TTFT / TPOT numbers | **no** |
| KV cache pressure, behavior under concurrency | **no** |
| AWQ Marlin kernel dispatch | **no** |
| `kv_cache_dtype=fp8`, `gpu_memory_utilization` semantics | **no** |

The "no" items are all CUDA-specific. They are exercised exactly once, on the rented RunPod GPU, after the rehearsal completes locally.

## Apple Silicon toolchain notes

- **Use native ARM64 Python 3.12.** Rosetta x86_64 Python is not supported by vllm-metal / vllm-mlx and is slower for everything else. `uv python install 3.12` on macOS gives you the right thing.
- **Docker Desktop for macOS is fine for building the CUDA image.** Building validates the Dockerfile is well-formed even without a GPU to run it.
- **The `vllm-project/vllm-metal` plugin** (Metal/MLX backend, <https://github.com/vllm-project/vllm-metal>) would give faster local inference, but it is *not* what runs on RunPod. Using it for smoke could mask CUDA-path bugs. Stick with vLLM CPU or Ollama.
