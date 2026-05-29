"""Tests for the lm-eval command builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.eval.config import EvalSuite
from forge.eval.runner import _build_command, run_suite


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


def test_command_uses_local_completions_model() -> None:
    cmd = _build_command("/lm-eval", _suite())
    assert cmd[0] == "/lm-eval"
    # local-completions supports both loglikelihood (MMLU, HellaSwag) and
    # generate_until (GSM8K). local-chat-completions does not.
    assert cmd[cmd.index("--model") + 1] == "local-completions"


def test_command_model_args_targets_completions_endpoint() -> None:
    cmd = _build_command("/lm-eval", _suite())
    model_args = cmd[cmd.index("--model_args") + 1]
    assert "base_url=http://localhost:8000/v1/completions" in model_args
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


def test_run_suite_accepts_valid_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suite = _suite(result_dir=tmp_path)

    class _FakeProc:
        def __init__(self) -> None:
            self.returncode = 0

    def fake_run(cmd: list[str], check: bool = False) -> _FakeProc:
        output_path = Path(cmd[cmd.index("--output_path") + 1])
        output_path.write_text(
            json.dumps({"results": {"hellaswag": {"acc_norm,none": 0.1}}}),
            encoding="utf-8",
        )
        return _FakeProc()

    monkeypatch.setattr("forge.eval.runner.subprocess.run", fake_run)

    outcome = run_suite(suite, lm_eval_executable="/fake/lm-eval")

    assert outcome.succeeded
    assert outcome.result_path == suite.result_path


def test_run_suite_rejects_result_missing_configured_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _suite(result_dir=tmp_path)

    class _FakeProc:
        def __init__(self) -> None:
            self.returncode = 0

    def fake_run(cmd: list[str], check: bool = False) -> _FakeProc:
        output_path = Path(cmd[cmd.index("--output_path") + 1])
        output_path.write_text(json.dumps({"results": {}}), encoding="utf-8")
        return _FakeProc()

    monkeypatch.setattr("forge.eval.runner.subprocess.run", fake_run)

    outcome = run_suite(suite, lm_eval_executable="/fake/lm-eval")

    assert not outcome.succeeded
