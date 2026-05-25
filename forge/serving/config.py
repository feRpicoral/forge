"""Env-driven serving config.

Resolves at process start. All values can be overridden via environment variables
(see `.env.example` for the canonical reference). CUDA-only fields are honored by
vLLM only when running on a CUDA build; on the macOS CPU backend they are silently
ignored by the engine itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

Quantization = Literal["none", "awq", "gptq"]
KVCacheDtype = Literal["auto", "fp8", "fp8_e5m2"]


def _env(name: str, default: str) -> str:
    """Read an env var with a fallback. Empty strings count as unset."""
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


@dataclass(frozen=True)
class ServingConfig:
    """Resolved serving configuration.

    Use ``ServingConfig.from_env()`` to construct from environment, or pass keyword
    arguments directly in tests.
    """

    model_id: str
    quantization: Quantization
    max_model_len: int
    max_num_seqs: int
    host: str
    port: int
    gpu_memory_utilization: float
    kv_cache_dtype: KVCacheDtype
    tensor_parallel_size: int
    hf_token: str | None = field(repr=False, default=None)
    hf_home: str | None = None

    @classmethod
    def from_env(cls) -> ServingConfig:
        """Build a config from environment variables.

        Raises ``ValueError`` for malformed numerics; returns sensible defaults
        otherwise. ``HF_TOKEN`` is stripped from ``repr`` to avoid accidental leaks.
        """
        quantization_raw = _env("QUANTIZATION", "none").lower()
        if quantization_raw not in ("none", "awq", "gptq"):
            raise ValueError(
                f"QUANTIZATION must be one of: none, awq, gptq — got {quantization_raw!r}"
            )

        kv_cache_dtype_raw = _env("KV_CACHE_DTYPE", "auto").lower()
        if kv_cache_dtype_raw not in ("auto", "fp8", "fp8_e5m2"):
            raise ValueError(
                f"KV_CACHE_DTYPE must be one of: auto, fp8, fp8_e5m2 — got {kv_cache_dtype_raw!r}"
            )

        return cls(
            model_id=_env("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"),
            quantization=quantization_raw,  # type: ignore[arg-type]
            max_model_len=_env_int("MAX_MODEL_LEN", 4096),
            max_num_seqs=_env_int("MAX_NUM_SEQS", 128),
            host=_env("SERVING_HOST", "0.0.0.0"),
            port=_env_int("SERVING_PORT", 8000),
            gpu_memory_utilization=_env_float("GPU_MEMORY_UTILIZATION", 0.90),
            kv_cache_dtype=kv_cache_dtype_raw,  # type: ignore[arg-type]
            tensor_parallel_size=_env_int("TENSOR_PARALLEL_SIZE", 1),
            hf_token=os.environ.get("HF_TOKEN") or None,
            hf_home=os.environ.get("HF_HOME") or None,
        )

    def to_vllm_args(self) -> list[str]:
        """Render the config as a ``vllm serve`` argv list.

        Returns a flat list suitable for ``subprocess.run`` or ``os.execvp``. Only
        flags with non-default vLLM behavior are emitted to keep the command line
        readable.
        """
        args: list[str] = [
            "vllm",
            "serve",
            self.model_id,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
        ]
        if self.quantization != "none":
            args.extend(["--quantization", self.quantization])
        if self.kv_cache_dtype != "auto":
            args.extend(["--kv-cache-dtype", self.kv_cache_dtype])
        return args

    @property
    def base_url(self) -> str:
        """The OpenAI-compatible base URL clients should target."""
        host = "localhost" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}/v1"
