"""Tests for the benchmark sweep YAML loader."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from forge.benchmark.config import BenchSweep, load_sweep


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "sweep.yaml"
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


def test_load_minimal_sweep(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        name: smoke
        description: Tiny smoke test
        model: Qwen/Qwen2.5-0.5B-Instruct
        base_url: http://localhost:8000/v1
        backend: openai-chat
        dataset:
          name: random
          num_prompts: 8
        seed: 42
        result_dir: results/bench/smoke
        concurrency_levels: [1, 2, 4]
        """,
    )
    sweep = load_sweep(path)

    assert sweep == BenchSweep(
        name="smoke",
        description="Tiny smoke test",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        base_url="http://localhost:8000/v1",
        backend="openai-chat",
        dataset=sweep.dataset,
        seed=42,
        result_dir=Path("results/bench/smoke"),
        concurrency_levels=[1, 2, 4],
    )
    assert sweep.dataset.name == "random"
    assert sweep.dataset.num_prompts == 8
    assert sweep.dataset.extra_args == {}


def test_load_with_dataset_extra_args(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        name: smoke
        description: Tiny smoke test
        model: m
        base_url: http://x/v1
        backend: openai-chat
        dataset:
          name: random
          num_prompts: 8
          extra_args:
            random-input-len: 128
            random-output-len: 64
        seed: 42
        result_dir: out
        concurrency_levels: [1]
        """,
    )
    sweep = load_sweep(path)
    assert sweep.dataset.extra_args == {"random-input-len": "128", "random-output-len": "64"}


def test_result_path_zero_padded(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        name: smoke
        description: x
        model: m
        base_url: http://x/v1
        backend: openai-chat
        dataset: {name: r, num_prompts: 1}
        seed: 1
        result_dir: out
        concurrency_levels: [1, 64]
        """,
    )
    sweep = load_sweep(path)
    assert sweep.result_path(1) == Path("out/smoke-c0001.json")
    assert sweep.result_path(64) == Path("out/smoke-c0064.json")


def test_rejects_missing_top_level(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        name: smoke
        description: x
        """,
    )
    with pytest.raises(TypeError, match="'dataset'"):
        load_sweep(path)


def test_rejects_empty_concurrency(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        name: smoke
        description: x
        model: m
        base_url: http://x
        backend: openai
        dataset: {name: r, num_prompts: 1}
        seed: 1
        result_dir: out
        concurrency_levels: []
        """,
    )
    with pytest.raises(ValueError, match="concurrency_levels must not be empty"):
        load_sweep(path)


def test_rejects_non_positive_concurrency(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        name: smoke
        description: x
        model: m
        base_url: http://x
        backend: openai
        dataset: {name: r, num_prompts: 1}
        seed: 1
        result_dir: out
        concurrency_levels: [1, 0]
        """,
    )
    with pytest.raises(ValueError, match="concurrency_levels\\[1\\] must be a positive int"):
        load_sweep(path)


def test_rejects_bool_for_int_field(tmp_path: Path) -> None:
    # YAML's `true` parses as bool. Bools are int subclass — guard against that.
    path = write(
        tmp_path,
        """
        name: smoke
        description: x
        model: m
        base_url: http://x
        backend: openai
        dataset: {name: r, num_prompts: true}
        seed: 1
        result_dir: out
        concurrency_levels: [1]
        """,
    )
    with pytest.raises(TypeError, match="'num_prompts'"):
        load_sweep(path)


def test_real_smoke_config_parses() -> None:
    """The repo's actual bench-smoke.yaml must always parse cleanly."""
    sweep = load_sweep(Path("configs/bench-smoke.yaml"))
    assert sweep.name == "smoke"
    assert sweep.dataset.name == "random"
    assert sweep.concurrency_levels == [1, 2]


def test_real_full_configs_parse() -> None:
    """Both production configs must always parse cleanly."""
    bf16 = load_sweep(Path("configs/bench-full.yaml"))
    awq = load_sweep(Path("configs/bench-full-awq.yaml"))
    dataset_path = "/workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

    assert bf16.name == "full-bf16"
    assert awq.name == "full-awq"
    assert bf16.concurrency_levels == awq.concurrency_levels
    assert bf16.dataset.num_prompts == awq.dataset.num_prompts
    assert bf16.dataset.extra_args["dataset_path"] == dataset_path
    assert awq.dataset.extra_args["dataset_path"] == dataset_path
