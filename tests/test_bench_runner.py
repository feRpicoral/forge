"""Tests for the benchmark runner command-builder logic.

The subprocess execution itself isn't unit-tested — it's exercised end-to-end
during smoke and pre-flight. The pure functions (URL normalization, argv
construction) are tested here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.benchmark.config import BenchSweep, DatasetConfig
from forge.benchmark.runner import _build_command, _normalize_base_url, run_sweep


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://localhost:8000/v1", "http://localhost:8000"),
        ("http://localhost:8000/v1/", "http://localhost:8000"),
        ("http://localhost:8000", "http://localhost:8000"),
        ("https://api.example.com:443/v1", "https://api.example.com:443"),
    ],
)
def test_normalize_base_url(base_url: str, expected: str) -> None:
    assert _normalize_base_url(base_url) == expected


def _sweep(**overrides: object) -> BenchSweep:
    defaults: dict[str, object] = {
        "name": "smoke",
        "description": "x",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "backend": "openai-chat",
        "dataset": DatasetConfig(name="random", num_prompts=8),
        "seed": 42,
        "result_dir": Path("results/bench/smoke"),
        "concurrency_levels": [1, 2],
    }
    defaults.update(overrides)
    return BenchSweep(**defaults)  # type: ignore[arg-type]


def test_build_command_minimal() -> None:
    cmd = _build_command("/path/to/vllm", _sweep(), concurrency=4)

    assert cmd[0] == "/path/to/vllm"
    assert cmd[1:3] == ["bench", "serve"]
    assert "--backend" in cmd
    assert cmd[cmd.index("--backend") + 1] == "openai-chat"
    assert cmd[cmd.index("--base-url") + 1] == "http://localhost:8000"
    assert cmd[cmd.index("--endpoint") + 1] == "/v1/chat/completions"
    assert cmd[cmd.index("--max-concurrency") + 1] == "4"
    assert "--save-result" in cmd
    assert cmd[cmd.index("--result-filename") + 1] == "smoke-c0004.json"


def test_build_command_openai_completions_endpoint() -> None:
    cmd = _build_command("/vllm", _sweep(backend="openai"), concurrency=1)
    assert cmd[cmd.index("--endpoint") + 1] == "/v1/completions"


def test_build_command_passes_dataset_extra_args() -> None:
    sweep = _sweep(
        dataset=DatasetConfig(
            name="random",
            num_prompts=8,
            extra_args={"random-input-len": "128", "random-output-len": "64"},
        )
    )
    cmd = _build_command("/vllm", sweep, concurrency=1)

    assert "--random-input-len" in cmd
    assert cmd[cmd.index("--random-input-len") + 1] == "128"
    assert "--random-output-len" in cmd
    assert cmd[cmd.index("--random-output-len") + 1] == "64"


def test_build_command_extra_args_underscore_converted_to_dash() -> None:
    sweep = _sweep(
        dataset=DatasetConfig(
            name="random",
            num_prompts=8,
            extra_args={"random_input_len": "128"},
        )
    )
    cmd = _build_command("/vllm", sweep, concurrency=1)
    assert "--random-input-len" in cmd
    assert "--random_input_len" not in cmd


def test_run_sweep_records_monotonic_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``duration_seconds`` must reflect the wall-clock cost of ``vllm bench``."""

    monotonic_values = iter([100.0, 102.5, 110.0, 114.25])

    def fake_monotonic() -> float:
        return next(monotonic_values)

    class _FakeProc:
        def __init__(self) -> None:
            self.returncode = 0

    def fake_run(cmd: list[str], check: bool = False) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr("forge.benchmark.runner.time.monotonic", fake_monotonic)
    monkeypatch.setattr("forge.benchmark.runner.subprocess.run", fake_run)

    outcomes = run_sweep(_sweep(result_dir=tmp_path), vllm_executable="/fake/vllm")

    assert [o.concurrency for o in outcomes] == [1, 2]
    assert outcomes[0].duration_seconds == pytest.approx(2.5)
    assert outcomes[1].duration_seconds == pytest.approx(4.25)
