# Forge — HuggingFace Spaces demo

Minimal Gradio app that lets a visitor send a prompt, see the response stream, and read the TTFT / mean TPOT / decode tokens-per-second computed client-side from the stream timestamps.

## Local development on M1

```bash
# Option A: against the local vLLM CPU server
OPENAI_BASE_URL=http://localhost:8000/v1 python deploy/hf-spaces/app.py

# Option B: against Ollama (faster on M1)
brew install ollama && ollama serve &
ollama pull qwen2.5:0.5b
OPENAI_BASE_URL=http://localhost:11434/v1 python deploy/hf-spaces/app.py
```

Then open <http://localhost:7860>.

## Deploying to HuggingFace Spaces

Two viable paths:

### A) Endpoint-driven (recommended for low ongoing cost)

The Space is a thin Gradio frontend that talks to a vLLM endpoint you host elsewhere (a RunPod pod, a cloud VM, etc.). Cheapest because the Space itself is free CPU — no GPU credits consumed.

1. Create a new Space:
   ```bash
   huggingface-cli login
   huggingface-cli repo create forge-demo --type space --space-sdk gradio
   ```
2. Add `app.py`, `requirements.txt`, and `README.md` (with the Spaces frontmatter — see below) to the Space's repo.
3. Set the Space's environment variables in the UI:
   - `OPENAI_BASE_URL=https://your-vllm-endpoint/v1`
   - `OPENAI_API_KEY=…` (if your endpoint requires auth)
   - `MODEL_ID=meta-llama/Llama-3.1-8B-Instruct` (optional — leave blank to autodetect via `/v1/models`)

### B) ZeroGPU (fully self-contained, time-sliced H100)

Load vLLM inside the Space and point `OPENAI_BASE_URL` at `http://localhost:8000/v1`. Each request gets a time-sliced H100; cold-start latency is real but acceptable for a demo. Requires the Space SDK to be set to "Docker" with a custom Dockerfile that installs `vllm` and launches both `vllm serve` and `python app.py`.

For simplicity, start with (A). Move to (B) only if no external endpoint is available.

## Spaces frontmatter

The Space's README needs this YAML block at the top to register itself with the Hub:

```yaml
---
title: Forge — Live LLM Latency
emoji: ⚡
colorFrom: blue
colorTo: orange
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
license: mit
---
```

## requirements.txt

The Space's `requirements.txt` needs:

```
gradio>=5.0
openai>=1.50
```

That's it — the demo is intentionally light. Heavy ML deps stay on the inference endpoint.
