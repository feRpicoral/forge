"""Subprocess wrapper around ``vllm bench serve``.

We invoke the vLLM CLI rather than calling its internal API: the CLI surface is
stable across versions in a way the Python API is not, and decoupling our code
from vLLM internals means a vLLM upgrade is a config change rather than a port.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from forge.benchmark.config import BenchSweep


@dataclass(frozen=True)
class RunOutcome:
    concurrency: int
    result_path: Path
    duration_seconds: float
    returncode: int

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


def run_sweep(sweep: BenchSweep, *, vllm_executable: str | None = None) -> list[RunOutcome]:
    """Run every concurrency level in the sweep against the configured server.

    The caller is responsible for ensuring the vLLM server at ``sweep.base_url``
    is up and serving ``sweep.model``. The harness does not start or stop the
    server — that is the orchestrator's job (locally: ``make serve`` in another
    shell; on RunPod: ``deploy/runpod-run.sh``).
    """
    vllm = vllm_executable or shutil.which("vllm")
    if not vllm:
        raise FileNotFoundError(
            "`vllm` not found on PATH. Install with `uv pip install -c constraints/serve.txt vllm`."
        )

    sweep.result_dir.mkdir(parents=True, exist_ok=True)

    outcomes: list[RunOutcome] = []
    for concurrency in sweep.concurrency_levels:
        result_path = sweep.result_path(concurrency)
        cmd = _build_command(vllm, sweep, concurrency)
        print(f"[forge] bench c={concurrency} → {result_path}", file=sys.stderr)
        print(f"[forge] exec: {' '.join(cmd)}", file=sys.stderr)

        proc = subprocess.run(cmd, check=False)
        outcomes.append(
            RunOutcome(
                concurrency=concurrency,
                result_path=result_path,
                duration_seconds=0.0,
                returncode=proc.returncode,
            )
        )
        if proc.returncode != 0:
            print(
                f"[forge] concurrency {concurrency} failed (rc={proc.returncode}); aborting sweep.",
                file=sys.stderr,
            )
            break

    return outcomes


_ENDPOINT_BY_BACKEND = {
    "vllm": "/v1/completions",
    "openai": "/v1/completions",
    "openai-chat": "/v1/chat/completions",
    "openai-audio": "/v1/audio/transcriptions",
    "openai-embeddings": "/v1/embeddings",
}


def _normalize_base_url(base_url: str) -> str:
    """Strip a trailing ``/v1`` so vllm bench's ``base_url + endpoint`` concat works.

    The serving layer reports its base URL as ``http://host:port/v1`` (matches
    the OpenAI client convention). ``vllm bench serve`` builds request URLs by
    concatenation, so it expects ``http://host:port`` — passing the ``/v1`` form
    yields a doubled path like ``/v1/v1/chat/completions`` and a 404.
    """
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    return trimmed


def _build_command(vllm: str, sweep: BenchSweep, concurrency: int) -> list[str]:
    endpoint = _ENDPOINT_BY_BACKEND.get(sweep.backend, "/v1/completions")
    cmd: list[str] = [
        vllm,
        "bench",
        "serve",
        "--backend",
        sweep.backend,
        "--base-url",
        _normalize_base_url(sweep.base_url),
        "--endpoint",
        endpoint,
        "--model",
        sweep.model,
        "--dataset-name",
        sweep.dataset.name,
        "--num-prompts",
        str(sweep.dataset.num_prompts),
        "--max-concurrency",
        str(concurrency),
        "--seed",
        str(sweep.seed),
        "--save-result",
        "--result-dir",
        str(sweep.result_dir),
        "--result-filename",
        sweep.result_filename(concurrency),
    ]
    for key, value in sweep.dataset.extra_args.items():
        cmd.extend([f"--{key.replace('_', '-')}", value])
    return cmd
