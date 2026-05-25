"""Tests for the lm-eval command builder."""

from __future__ import annotations

from pathlib import Path

from forge.eval.config import EvalSuite
from forge.eval.runner import _build_command


def _suite(**overrides: object) -> EvalSuite:
    defaults: dict[str, object] = {
        "name": "smoke",
        "description": "x",
        "variant": "smoke",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "tasks": ["hellaswag"],
        "num_fewshot": 0,
        "batch_size": 1,
        "limit": 4,
        "seed": 42,
        "result_dir": Path("results/eval/smoke"),
    }
    defaults.update(overrides)
    return EvalSuite(**defaults)  # type: ignore[arg-type]


def test_command_uses_local_chat_completions_model() -> None:
    cmd = _build_command("/lm-eval", _suite())
    assert cmd[0] == "/lm-eval"
    assert cmd[cmd.index("--model") + 1] == "local-chat-completions"


def test_command_model_args_targets_chat_completions_endpoint() -> None:
    cmd = _build_command("/lm-eval", _suite())
    model_args = cmd[cmd.index("--model_args") + 1]
    assert "base_url=http://localhost:8000/v1/chat/completions" in model_args
    assert "model=Qwen/Qwen2.5-0.5B-Instruct" in model_args


def test_command_joins_tasks() -> None:
    cmd = _build_command("/lm-eval", _suite(tasks=["mmlu", "gsm8k", "hellaswag"]))
    assert cmd[cmd.index("--tasks") + 1] == "mmlu,gsm8k,hellaswag"


def test_command_includes_limit_only_when_set() -> None:
    with_limit = _build_command("/lm-eval", _suite(limit=4))
    assert "--limit" in with_limit and with_limit[with_limit.index("--limit") + 1] == "4"

    without_limit = _build_command("/lm-eval", _suite(limit=None))
    assert "--limit" not in without_limit


def test_command_passes_few_shot_and_batch_size() -> None:
    cmd = _build_command("/lm-eval", _suite(num_fewshot=5, batch_size=8))
    assert cmd[cmd.index("--num_fewshot") + 1] == "5"
    assert cmd[cmd.index("--batch_size") + 1] == "8"
