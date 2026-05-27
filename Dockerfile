# Production CUDA image — runs on RunPod / any NVIDIA host.
#
# Layers Forge onto vllm/vllm-openai:latest. The base image owns the
# vllm + torch + CUDA toolchain; we add our package and enforce the
# transformers pin from constraints/serve.txt. We deliberately do NOT
# create a second uv-managed venv — that would re-download the entire
# CUDA stack (~3 GB) on every build and is fragile to network blips.

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

# Install locked runtime deps into the base image's Python, then add Forge
# without re-resolving dependencies. The base image already owns vLLM/CUDA.
RUN uv export --quiet \
      --frozen \
      --no-dev \
      --no-emit-project \
      --format requirements.txt \
      --output-file /tmp/forge-requirements.txt \
 && uv pip install --system -r /tmp/forge-requirements.txt \
 && rm /tmp/forge-requirements.txt \
 && uv pip install --system --no-deps . \
 && uv pip install --system -c constraints/serve.txt transformers

EXPOSE 8000

# The base image defines an ENTRYPOINT we want to override.
ENTRYPOINT []
CMD ["python", "-m", "scripts.serve"]
