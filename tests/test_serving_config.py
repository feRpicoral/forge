"""Tests for the env-driven ServingConfig."""

from __future__ import annotations

import pytest

from forge.serving.config import ServingConfig


def test_from_env_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "MODEL_ID",
        "QUANTIZATION",
        "MAX_MODEL_LEN",
        "MAX_NUM_SEQS",
        "SERVING_HOST",
        "SERVING_PORT",
        "GPU_MEMORY_UTILIZATION",
        "KV_CACHE_DTYPE",
        "TENSOR_PARALLEL_SIZE",
        "HF_TOKEN",
        "HF_HOME",
    ]:
        monkeypatch.delenv(key, raising=False)

    config = ServingConfig.from_env()

    assert config.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.quantization == "none"
    assert config.max_model_len == 4096
    assert config.max_num_seqs == 128
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.gpu_memory_utilization == pytest.approx(0.90)
    assert config.kv_cache_dtype == "auto"
    assert config.tensor_parallel_size == 1
    assert config.hf_token is None


def test_from_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
    monkeypatch.setenv("QUANTIZATION", "AWQ")
    monkeypatch.setenv("MAX_MODEL_LEN", "8192")
    monkeypatch.setenv("MAX_NUM_SEQS", "64")
    monkeypatch.setenv("SERVING_PORT", "8888")
    monkeypatch.setenv("GPU_MEMORY_UTILIZATION", "0.85")
    monkeypatch.setenv("KV_CACHE_DTYPE", "fp8")
    monkeypatch.setenv("TENSOR_PARALLEL_SIZE", "2")
    monkeypatch.setenv("HF_TOKEN", "hf_secret")

    config = ServingConfig.from_env()

    assert config.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert config.quantization == "awq"
    assert config.max_model_len == 8192
    assert config.max_num_seqs == 64
    assert config.port == 8888
    assert config.gpu_memory_utilization == pytest.approx(0.85)
    assert config.kv_cache_dtype == "fp8"
    assert config.tensor_parallel_size == 2
    assert config.hf_token == "hf_secret"


def test_from_env_rejects_unknown_quantization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANTIZATION", "fp4")
    with pytest.raises(ValueError, match="QUANTIZATION must be one of"):
        ServingConfig.from_env()


def test_from_env_rejects_unknown_kv_cache_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KV_CACHE_DTYPE", "int8")
    with pytest.raises(ValueError, match="KV_CACHE_DTYPE must be one of"):
        ServingConfig.from_env()


def test_from_env_rejects_non_numeric(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_MODEL_LEN", "lots")
    with pytest.raises(ValueError, match="MAX_MODEL_LEN must be an integer"):
        ServingConfig.from_env()


def test_to_vllm_args_baseline() -> None:
    config = ServingConfig(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        quantization="none",
        max_model_len=2048,
        max_num_seqs=4,
        host="0.0.0.0",
        port=8000,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="auto",
        tensor_parallel_size=1,
    )

    args = config.to_vllm_args()

    assert args[0:3] == ["vllm", "serve", "Qwen/Qwen2.5-0.5B-Instruct"]
    assert "--max-model-len" in args
    assert "2048" in args
    assert "--quantization" not in args
    assert "--kv-cache-dtype" not in args


def test_to_vllm_args_with_awq_and_fp8() -> None:
    config = ServingConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        quantization="awq",
        max_model_len=4096,
        max_num_seqs=128,
        host="0.0.0.0",
        port=8000,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        tensor_parallel_size=1,
    )

    args = config.to_vllm_args()

    assert "--quantization" in args
    quant_idx = args.index("--quantization")
    assert args[quant_idx + 1] == "awq"

    assert "--kv-cache-dtype" in args
    kv_idx = args.index("--kv-cache-dtype")
    assert args[kv_idx + 1] == "fp8"


def test_base_url_localhost_replacement() -> None:
    config = ServingConfig(
        model_id="x",
        quantization="none",
        max_model_len=1,
        max_num_seqs=1,
        host="0.0.0.0",
        port=8000,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="auto",
        tensor_parallel_size=1,
    )
    assert config.base_url == "http://localhost:8000/v1"


def test_base_url_preserves_explicit_host() -> None:
    config = ServingConfig(
        model_id="x",
        quantization="none",
        max_model_len=1,
        max_num_seqs=1,
        host="vllm.internal",
        port=9000,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="auto",
        tensor_parallel_size=1,
    )
    assert config.base_url == "http://vllm.internal:9000/v1"


def test_repr_hides_hf_token() -> None:
    config = ServingConfig(
        model_id="x",
        quantization="none",
        max_model_len=1,
        max_num_seqs=1,
        host="0.0.0.0",
        port=8000,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="auto",
        tensor_parallel_size=1,
        hf_token="hf_super_secret",
    )
    rendered = repr(config)
    assert "hf_super_secret" not in rendered
