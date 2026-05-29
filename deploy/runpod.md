# RunPod deployment guide

This document is the reproduction recipe for the Forge benchmark and the
authoritative pre-flight checklist that protects the GPU budget. Every paid
GPU run must pass every box in this checklist *before* the pod is started.

## Why this exists

The plan budgets $50–$100 total for paid GPU time, with $0.27/hr as the active
RTX A5000 pod's compute rate on 2026-05-29. Storage adds about $0.01/hr, bringing
the active running pod to roughly $0.28/hr. That gives a working margin of
roughly 175–350 pod-hours total — plenty for this project, but still worth
protecting from config typos and missing tokens. The orchestrator script
(`./runpod-run.sh`) and this checklist together keep the run to one
tightly-scripted shell command, not exploration.

## Choosing the pod

| Setting | Value | Why |
|---|---|---|
| GPU | **RTX A5000 24 GB** | Available 24 GB tier that fits Llama 3.1 8B BF16 + KV cache headroom. |
| Image | `runpod-torch-v240` | Active pod image. Matches the PyTorch/CUDA runtime needed by vLLM. |
| vCPU / memory | **9 vCPU / 50 GB RAM** | Active pod shape. |
| Container disk | **20 GB** | Keep model weights off this disk. |
| Volume | **50 GB mounted at `/workspace`** | Llama 3.1 8B BF16 weights ~16 GB + AWQ ~5 GB + caches + logs. |
| Price | **$0.27/hr compute; ~$0.28/hr running total** | $0.003/hr container storage + $0.007/hr volume storage. |
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
- [ ] `configs/bench-full.yaml` and `configs/bench-full-awq.yaml`: model id (correct checkpoint per variant), `num_prompts`, `concurrency_levels`, `result_dir`, `seed`, `dataset.extra_args.dataset_path`.
- [ ] `configs/eval-full-bf16.yaml` and `configs/eval-full-awq.yaml`: tasks list, `num_fewshot`, `batch_size`, `limit` (must be `null` for the real run), `result_dir`.

### Hugging Face access

- [ ] `HF_TOKEN` env var is set on the pod's environment template, or you will log in once on the pod using the snippet below.
- [ ] Token has "read" scope and is approved for the gated Llama 3.1 8B repo.
- [ ] The pod either has `HF_TOKEN` set or has a cached Hugging Face token under `HF_HOME`.
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
- [ ] Written down: target total spend ≤ $1.25 (~$0.28/hr running total × ~4 hours including setup).
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
export HF_HOME=/workspace/.cache/huggingface
mkdir -p "$HF_HOME"
read -rsp "HF token: " HF_TOKEN
echo
export HF_TOKEN
uv run python - <<'PY'
import os
from huggingface_hub import HfApi, login

token = os.environ["HF_TOKEN"]
login(token=token, add_to_git_credential=False)
api = HfApi(token=token)
print("user:", api.whoami()["name"])
print("model:", api.model_info("meta-llama/Llama-3.1-8B-Instruct").id)
PY
unset HF_TOKEN

# 3. Download the ShareGPT trace
mkdir -p /workspace/datasets
curl -fL --retry 5 --continue-at - \
  -o /workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# 4. Run BF16 variant detached from SSH
mkdir -p logs
nohup bash deploy/runpod-run.sh --variant bf16 > logs/runpod-bf16.log 2>&1 &
tail -f logs/runpod-bf16.log

# 5. Run AWQ variant detached from SSH
nohup bash deploy/runpod-run.sh --variant awq > logs/runpod-awq.log 2>&1 &
tail -f logs/runpod-awq.log

# 6. Pull results to local
# (from your M1, or use rsync / pod's file browser)
```

Do not run the paid variants in the SSH foreground. If the SSH connection drops,
the foreground shell can terminate the orchestrator before it reaches eval.

If a benchmark sweep completed but eval did not, resume only the missing eval:

```bash
nohup bash deploy/runpod-run.sh --variant bf16 --only-eval > logs/runpod-bf16-eval.log 2>&1 &
tail -f logs/runpod-bf16-eval.log
```

## Result validation

`scripts.bench` validates each expected `vllm bench` JSON after the subprocess
exits. `scripts.eval` validates that the lm-eval output is parseable and
contains every configured task before reporting success.

To verify already-written files without rerunning the GPU work:

```bash
uv run python -m scripts.bench --config configs/bench-full.yaml --verify-only
uv run python -m scripts.eval --config configs/eval-full-bf16.yaml --verify-only
uv run python -m scripts.bench --config configs/bench-full-awq.yaml --verify-only
uv run python -m scripts.eval --config configs/eval-full-awq.yaml --verify-only
```

## After the run

- Stop the pod immediately. RunPod bills by the second.
- Pull `results/bench/full-bf16/`, `results/bench/full-awq/`, and `results/eval/full/` to local.
- Run `make chart` locally to regenerate every chart from the new data.
- Reconcile spend against the estimate before updating the README cost claim.
- Commit the new `results/` JSONs and regenerated charts to a `chore(results): add runpod-<date>` commit.
