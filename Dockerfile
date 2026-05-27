# Production CUDA image — runs on RunPod / any NVIDIA host.
#
# Layers Forge onto vllm/vllm-openai:latest. The base image owns the
# vllm + torch + CUDA toolchain; we add our package and enforce the
# transformers pin from constraints/serve.txt. We deliberately do NOT
# create a second uv-managed venv — that would re-download the entire
# CUDA stack (~3 GB) on every build and is fragile to network blips.
#
# Tag is intentionally floating to `latest` during development; before the
# Phase 7 paid run, this gets pinned to a specific vLLM release and the SHA
# is recorded in DECISIONS.md.

FROM vllm/vllm-openai:latest

LABEL org.opencontainers.image.title="Forge"
LABEL org.opencontainers.image.description="Production-grade LLM inference serving & optimization"
LABEL org.opencontainers.image.source="https://github.com/feRpicoral/forge"
LABEL org.opencontainers.image.licenses="MIT"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY forge/ ./forge/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY constraints/ ./constraints/

# Install Forge + the small runtime deps from pyproject (pyyaml, matplotlib,
# etc.) into the base image's Python. ``--system`` avoids creating a second
# venv that would re-download vllm + xformers + the nvidia/* wheels — the
# base image already owns all of those.
# Then enforce the transformers pin via the constraint file. No-op when the
# base image's transformers already satisfies it.
RUN uv pip install --system . \
 && uv pip install --system -c constraints/serve.txt transformers

EXPOSE 8000

# The base image defines an ENTRYPOINT we want to override.
ENTRYPOINT []
CMD ["python", "-m", "scripts.serve"]
