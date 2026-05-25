"""Tests for AWQ quantization config + CLI argument parsing.

The actual quantization is not unit-tested — it requires a GPU and downloading
~16 GB of weights. The CLI's job is to parse args correctly and the config's
job is to render the right ``quant_config`` dict for AutoAWQ.
"""

from __future__ import annotations

from pathlib import Path

from forge.quantization.awq import AWQConfig
from scripts.quantize import build_config


def test_default_config_matches_canonical_recipe() -> None:
    config = AWQConfig()
    assert config.w_bit == 4
    assert config.q_group_size == 128
    assert config.zero_point is True
    assert config.version == "GEMM"
    assert config.source_model == "meta-llama/Llama-3.1-8B-Instruct"


def test_quant_config_shape_matches_autoawq() -> None:
    """AutoAWQForCausalLM.quantize expects exactly these four keys."""
    config = AWQConfig()
    assert set(config.quant_config.keys()) == {"w_bit", "q_group_size", "zero_point", "version"}


def test_describe_includes_key_params() -> None:
    config = AWQConfig()
    described = config.describe()
    assert "w_bit=4" in described
    assert "q_group_size=128" in described
    assert "GEMM" in described
    assert "meta-llama/Llama-3.1-8B-Instruct" in described


def test_cli_default_args() -> None:
    config, dry_run = build_config([])
    assert config.source_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert config.w_bit == 4
    assert config.q_group_size == 128
    assert config.zero_point is True
    assert config.version == "GEMM"
    assert dry_run is False


def test_cli_overrides() -> None:
    config, dry_run = build_config(
        [
            "--source",
            "Qwen/Qwen2.5-7B-Instruct",
            "--output",
            "/tmp/qwen-awq",
            "--w-bit",
            "3",
            "--q-group-size",
            "64",
            "--no-zero-point",
            "--version",
            "GEMV",
        ]
    )
    assert config.source_model == "Qwen/Qwen2.5-7B-Instruct"
    assert config.output_path == Path("/tmp/qwen-awq")
    assert config.w_bit == 3
    assert config.q_group_size == 64
    assert config.zero_point is False
    assert config.version == "GEMV"
    assert dry_run is False


def test_cli_dry_run_flag() -> None:
    _, dry_run = build_config(["--dry-run"])
    assert dry_run is True


def test_describe_emits_no_zero_point_correctly() -> None:
    config = AWQConfig(zero_point=False)
    assert "zero_point=False" in config.describe()
