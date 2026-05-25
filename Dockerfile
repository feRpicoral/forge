# Production CUDA image — runs on RunPod / any NVIDIA host.
#
# Base: the vLLM project's official image. They pin the torch/CUDA/transformers
# triple themselves, which is exactly the matrix we don't want to fight. We layer
# our package on top.
#
# Tag is intentionally floating to `latest` during development; before the Phase 7
# paid run, this gets pinned to a specific vLLM release and the SHA is recorded
# in DECISIONS.md.

FROM vllm/vllm-openai:latest

# Runtime metadata
LABEL org.opencontainers.image.title="Forge"
LABEL org.opencontainers.image.description="Production-grade LLM inference serving & optimization"
LABEL org.opencontainers.image.source="https://github.com/feRpicoral/forge"
LABEL org.opencontainers.image.licenses="MIT"

# uv for fast, reproducible installs inside the container.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install Python deps first for layer cache friendliness.
COPY pyproject.toml uv.lock README.md ./
COPY forge/ ./forge/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY constraints/ ./constraints/

RUN uv sync --frozen --no-dev

# vLLM is pre-installed in the base image, but we re-install against our
# constraints file to enforce the transformers pin. Safe no-op when versions
# already match.
RUN uv pip install -c constraints/serve.txt vllm transformers

ENV PATH="/app/.venv/bin:${PATH}"

# vLLM exposes 8000 by default; the entrypoint script reads SERVING_PORT.
EXPOSE 8000

# The vllm/vllm-openai base image defines an ENTRYPOINT we want to override.
ENTRYPOINT []
CMD ["python", "-m", "scripts.serve"]
