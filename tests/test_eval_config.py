"""Tests for the eval suite YAML loader."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from forge.eval.config import EvalSuite, load_suite


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


def test_load_minimal_suite(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        name: full
        description: x
        variant: bf16
        model: meta-llama/Llama-3.1-8B-Instruct
        base_url: http://localhost:8000/v1
        tasks: [mmlu, gsm8k, hellaswag]
        num_fewshot: 5
        batch_size: 4
        limit: null
        seed: 42
        result_dir: results/eval/full
        """,
    )
    suite = load_suite(path)
    assert suite == EvalSuite(
        name="full",
        description="x",
        variant="bf16",
        model="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:8000/v1",
        tasks=["mmlu", "gsm8k", "hellaswag"],
        num_fewshot=5,
        batch_size=4,
        limit=None,
        seed=42,
        result_dir=Path("results/eval/full"),
    )


def test_load_with_limit(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        name: smoke
        description: x
        variant: smoke
        model: m
        base_url: http://x
        tasks: [hellaswag]
        num_fewshot: 0
        batch_size: 1
        limit: 4
        seed: 42
        result_dir: out
        """,
    )
    assert load_suite(path).limit == 4


def test_rejects_zero_limit(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        name: smoke
        description: x
        variant: smoke
        model: m
        base_url: http://x
        tasks: [hellaswag]
        num_fewshot: 0
        batch_size: 1
        limit: 0
        seed: 42
        result_dir: out
        """,
    )
    with pytest.raises(TypeError, match="'limit'"):
        load_suite(path)


def test_rejects_empty_tasks(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        name: smoke
        description: x
        variant: smoke
        model: m
        base_url: http://x
        tasks: []
        num_fewshot: 0
        batch_size: 1
        limit: null
        seed: 42
        result_dir: out
        """,
    )
    with pytest.raises(TypeError, match="'tasks'"):
        load_suite(path)


def test_result_path_includes_variant() -> None:
    path = Path("configs/eval-full-bf16.yaml")
    suite = load_suite(path)
    assert suite.result_path == Path("results/eval/full/full-bf16.json")


def test_real_configs_parse() -> None:
    for cfg in ("eval-smoke.yaml", "eval-full-bf16.yaml", "eval-full-awq.yaml"):
        suite = load_suite(Path("configs") / cfg)
        assert suite.tasks
