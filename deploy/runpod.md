# RunPod deployment guide

This document is the reproduction recipe for the Forge benchmark and the
authoritative pre-flight checklist that protects the GPU budget. Every paid
GPU run must pass every box in this checklist *before* the pod is started.

## Why this exists

The plan budgets $50–$100 total for paid GPU time, with $0.69/hr as the listed
RTX 4090 Community Pod rate on 2026-05-29. That gives a working margin of
roughly 70–145 GPU-hours total — plenty for this project, but still worth
protecting from config typos and missing tokens. The orchestrator script
(`./runpod-run.sh`) and this checklist together keep the run to one
tightly-scripted shell command, not exploration.

## Choosing the pod

| Setting | Value | Why |
|---|---|---|
| GPU | **RTX 4090 24 GB (Community)** | Cheapest tier that fits Llama 3.1 8B BF16 + KV cache headroom. ~$0.69/hr. |
| Image | `nvidia/cuda:12.4.x-runtime-ubuntu22.04` or RunPod's "PyTorch 2.4 / CUDA 12.4" template | Matches vLLM's supported CUDA matrix. |
| Storage | **50 GB** persistent volume | Llama 3.1 8B BF16 weights ~16 GB + AWQ ~5 GB + caches + logs. Don't run out mid-bench. |
| Idle shutdown | **15 min** | Belt and suspenders against forgetting to stop the pod. |

## Pre-flight checklist

Every box must be ticked **on the M1**, on the latest commit, before the pod
is started. None of these requires the GPU.

### Repo + tooling

- [ ] `git status` shows a clean working tree on the branch you want to benchmark.
- [ ] `uv sync --frozen --group dev` reproduces the project lock cleanly.
- [ ] `uv pip install -c constraints/serve.txt vllm` succeeds and `uv run python -c "import vllm; print(vllm.__version__)"` prints the pinned version.
- [ ] `uv pip install -c constraints/eval.txt "lm-eval[api]"` succeeds (or pre-cache on the pod).
- [ ] All CI checks green on the latest commit (`gh pr checks` or the badge on the PR).

### Local validation

- [ ] `uv run ruff check .` exits 0.
- [ ] `uv run ruff format --check .` exits 0.
- [ ] `uv run mypy` exits 0.
- [ ] `uv run pytest` — all tests green.

### Config review (read line-by-line, don't just glance)

- [ ] `configs/server.yaml`: model id, max_model_len, max_num_seqs, kv_cache_dtype, gpu_memory_utilization, tensor_parallel_size.
- [ ] `configs/bench-full.yaml` and `configs/bench-full-awq.yaml`: model id (correct checkpoint per variant), `num_prompts`, `concurrency_levels`, `result_dir`, `seed`.
- [ ] `configs/eval-full-bf16.yaml` and `configs/eval-full-awq.yaml`: tasks list, `num_fewshot`, `batch_size`, `limit` (must be `null` for the real run), `result_dir`.

### Hugging Face access

- [ ] `HF_TOKEN` env var is set on the pod's environment template (NOT in committed code).
- [ ] Token has "read" scope and is approved for the gated Llama 3.1 8B repo.
- [ ] On the M1, pulling the *tokenizer only* succeeds:
  ```bash
  HF_TOKEN=$YOUR_TOKEN uv run python -c "
  from huggingface_hub import hf_hub_download
  print(hf_hub_download('meta-llama/Llama-3.1-8B-Instruct', 'tokenizer.json'))
  "
  ```
  If this works on M1, the full weight download will work on the pod.

### Rehearsal

- [ ] **`bash deploy/runpod-run.sh --rehearsal` completes end-to-end on M1 with no manual intervention.**
  This is the gate. The rehearsal starts vLLM CPU against `Qwen/Qwen2.5-0.5B-Instruct`, waits for `/health`, runs `bench-smoke` and `eval-smoke`, and stops the server. If that doesn't work locally, it won't work on the pod either — and you'd be paying for the discovery.

### Time + spend budget

- [ ] Written down: expected wall-clock for each variant (rough estimate: BF16 ~60 min, AWQ ~60 min).
- [ ] Written down: target total spend ≤ $3.00 ($0.69/hr × ~4 hours including setup).
- [ ] Hard abort plan: if the run blows past 2x the time estimate, `Ctrl-C` and re-evaluate before re-running.

## On-pod execution

After SSH-ing into the pod:

```bash
# 1. Clone and set up
git clone https://github.com/feRpicoral/forge.git
cd forge
uv sync --frozen --group dev
uv pip install -c constraints/serve.txt vllm
uv pip install -c constraints/eval.txt "lm-eval[api]"

# 2. Auth
export HF_TOKEN=hf_your_token_here
export HF_HOME=/workspace/.cache/huggingface

# 3. Run BF16 variant
bash deploy/runpod-run.sh --variant bf16

# 4. Run AWQ variant (server restart is handled by the script)
bash deploy/runpod-run.sh --variant awq

# 5. Pull results to local
# (from your M1, or use rsync / pod's file browser)
```

## After the run

- Stop the pod immediately. RunPod bills by the second.
- Pull `results/bench/full-bf16/`, `results/bench/full-awq/`, and `results/eval/full/` to local.
- Run `make chart` locally to regenerate every chart from the new data.
- Reconcile spend against the estimate before updating the README cost claim.
- Commit the new `results/` JSONs and regenerated charts to a `chore(results): add runpod-<date>` commit.
