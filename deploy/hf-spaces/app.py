"""Forge — live latency demo (HuggingFace Spaces).

Single textarea, one send button, a streaming completion panel, and a live
metrics readout (TTFT, mean TPOT, output tokens, tokens/sec). Talks to any
OpenAI-compatible endpoint configured via ``OPENAI_BASE_URL`` and
``OPENAI_API_KEY`` env vars.

Designed to deploy as-is to a free HuggingFace Space targeting a vLLM endpoint
the operator hosts elsewhere. For a fully self-contained ZeroGPU variant,
load vLLM into the Space and point ``OPENAI_BASE_URL`` at the in-process
server — out of scope for this minimal demo.

Local dev:
    OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama
    OPENAI_BASE_URL=http://localhost:8000/v1    # vLLM CPU
    python deploy/hf-spaces/app.py
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from dataclasses import dataclass, field

import gradio as gr
from openai import OpenAI

# Read at module load so the Space's env block is the single source of truth.
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "not-used-by-vllm")
MODEL_ID = os.environ.get("MODEL_ID", "")  # autodetect from /v1/models if empty
REPO_URL = "https://github.com/feRpicoral/forge"


@dataclass
class StreamMetrics:
    """Chunk-level timing captured during a single streaming completion.

    OpenAI-compatible streaming chunks aren't guaranteed to be one token each —
    a chunk can carry partial text or several tokens depending on how the
    server batches its SSE writes. Real token counts come from the optional
    ``usage`` payload at end-of-stream; until that arrives we report
    chunk-level numbers and label them as such.
    """

    request_started_at: float
    first_chunk_at: float | None = None
    last_chunk_at: float | None = None
    output_chunks: int = 0
    chunk_arrival_times: list[float] = field(default_factory=list)
    completion_tokens: int | None = None  # populated from final usage chunk if available

    @property
    def ttft_ms(self) -> float | None:
        """Time to first streamed content chunk. Proxies token-level TTFT — the
        first chunk carries at least one token, so this is an upper bound."""
        if self.first_chunk_at is None:
            return None
        return (self.first_chunk_at - self.request_started_at) * 1000.0

    @property
    def mean_chunk_interval_ms(self) -> float | None:
        if self.first_chunk_at is None or self.last_chunk_at is None or self.output_chunks < 2:
            return None
        decode_seconds = self.last_chunk_at - self.first_chunk_at
        return (decode_seconds / max(self.output_chunks - 1, 1)) * 1000.0

    @property
    def chunks_per_second(self) -> float | None:
        if self.first_chunk_at is None or self.last_chunk_at is None:
            return None
        decode_seconds = self.last_chunk_at - self.first_chunk_at
        if decode_seconds <= 0:
            return None
        return (self.output_chunks - 1) / decode_seconds

    @property
    def mean_tpot_ms(self) -> float | None:
        """Real per-token decode latency. Available only after usage arrives."""
        if (
            self.first_chunk_at is None
            or self.last_chunk_at is None
            or self.completion_tokens is None
            or self.completion_tokens < 2
        ):
            return None
        decode_seconds = self.last_chunk_at - self.first_chunk_at
        return (decode_seconds / (self.completion_tokens - 1)) * 1000.0

    @property
    def tokens_per_second(self) -> float | None:
        if (
            self.first_chunk_at is None
            or self.last_chunk_at is None
            or self.completion_tokens is None
        ):
            return None
        decode_seconds = self.last_chunk_at - self.first_chunk_at
        if decode_seconds <= 0:
            return None
        return (self.completion_tokens - 1) / decode_seconds

    def format(self) -> str:
        if self.completion_tokens is not None:
            token_rows = [
                f"**Output tokens**: {self.completion_tokens}",
                (
                    f"**Mean TPOT**: {self.mean_tpot_ms:.1f} ms"
                    if self.mean_tpot_ms is not None
                    else "**Mean TPOT**: …"
                ),
                (
                    f"**Tokens / sec (decode)**: {self.tokens_per_second:.1f}"
                    if self.tokens_per_second is not None
                    else "**Tokens / sec (decode)**: …"
                ),
            ]
        else:
            token_rows = [
                f"**Streamed chunks**: {self.output_chunks}",
                (
                    f"**Mean inter-chunk latency**: {self.mean_chunk_interval_ms:.1f} ms"
                    if self.mean_chunk_interval_ms is not None
                    else "**Mean inter-chunk latency**: …"
                ),
                (
                    f"**Chunks / sec (decode)**: {self.chunks_per_second:.1f}"
                    if self.chunks_per_second is not None
                    else "**Chunks / sec (decode)**: …"
                ),
            ]
        rows = [
            f"**TTFT**: {self.ttft_ms:.0f} ms" if self.ttft_ms is not None else "**TTFT**: …",
            *token_rows,
        ]
        return "  \n".join(rows)


def _resolve_model_id(client: OpenAI) -> str:
    if MODEL_ID:
        return MODEL_ID
    try:
        models = client.models.list().data
    except Exception:
        # Network surface — fall back to an empty string so the UI can show a
        # helpful error rather than crashing.
        return ""
    if not models:
        return ""
    return models[0].id


def stream_completion(
    prompt: str, max_tokens: int, temperature: float
) -> Generator[tuple[str, str], None, None]:
    """Yield ``(completion_text, metrics_md)`` as the model streams its response."""
    if not prompt.strip():
        yield "_(empty prompt)_", "Send a prompt to see the timing breakdown."
        return

    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY, timeout=120.0)
    model = _resolve_model_id(client)
    if not model:
        yield (
            "_(no model registered on the endpoint)_",
            f"Configure `OPENAI_BASE_URL` (currently `{OPENAI_BASE_URL}`) and ensure the server is up.",
        )
        return

    metrics = StreamMetrics(request_started_at=time.monotonic())

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            # vLLM honors include_usage; older servers ignore it gracefully. The
            # final chunk carries real prompt/completion token counts which we
            # use to switch from chunk-level to token-level metrics in the UI.
            stream_options={"include_usage": True},
        )
    except Exception as exc:
        # Surface any network or API error to the user as readable Markdown.
        yield (
            "",
            f"**error**: {type(exc).__name__}: {exc}\n\n"
            f"Endpoint: `{OPENAI_BASE_URL}` — model: `{model}`",
        )
        return

    buffer: list[str] = []
    for chunk in response:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            metrics.completion_tokens = getattr(usage, "completion_tokens", None)

        delta = chunk.choices[0].delta if chunk.choices else None
        content = getattr(delta, "content", None) if delta is not None else None
        if not content:
            continue

        now = time.monotonic()
        if metrics.first_chunk_at is None:
            metrics.first_chunk_at = now
        metrics.last_chunk_at = now
        metrics.output_chunks += 1
        metrics.chunk_arrival_times.append(now)
        buffer.append(content)
        yield "".join(buffer), metrics.format()

    # If the final usage chunk arrived after the last content chunk, the
    # display still reads chunk-level numbers — re-emit with token metrics now.
    if metrics.completion_tokens is not None:
        yield "".join(buffer) if buffer else "_(no content)_", metrics.format()
    elif not buffer:
        yield "_(no content)_", metrics.format()


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Forge — Live LLM Latency", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Forge — Live LLM Latency Demo\n\n"
            "Send a prompt and watch the response stream token by token, with TTFT, "
            f"mean TPOT, and decode tokens/sec captured client-side. [Source]({REPO_URL})."
        )
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Write a 4-line poem about a small GPU at midnight.",
                    lines=5,
                )
                with gr.Row():
                    max_tokens = gr.Slider(8, 1024, value=256, step=8, label="Max tokens")
                    temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                submit = gr.Button("Send", variant="primary")
            with gr.Column(scale=3):
                completion = gr.Markdown(label="Completion", value="_(awaiting prompt)_")
                metrics = gr.Markdown(
                    label="Live metrics",
                    value=(
                        "**TTFT**: …  \n"
                        "**Streamed chunks**: 0  \n"
                        "**Mean inter-chunk latency**: …  \n"
                        "**Chunks / sec (decode)**: …"
                    ),
                )

        submit.click(
            fn=stream_completion,
            inputs=[prompt, max_tokens, temperature],
            outputs=[completion, metrics],
        )

        gr.Markdown(
            f"---\n"
            f"Endpoint: `{OPENAI_BASE_URL}` — metrics are wall-clock client-side. "
            f"During streaming we report **chunk-level** numbers (OpenAI SSE chunks "
            f"aren't guaranteed to be one token); when the endpoint returns a final "
            f"`usage` block the display switches to real token counts and TPOT. "
            f"For full reproducibility (hardware, model SHA, vLLM version) see the [repo]({REPO_URL})."
        )

    return demo


if __name__ == "__main__":
    build_interface().launch(server_name="0.0.0.0", server_port=7860)
