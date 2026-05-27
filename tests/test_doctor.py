"""Tests for the env-drift diagnostic."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest
from packaging.version import Version

from scripts.doctor import (
    Check,
    check_constraint_file,
    check_executable,
    check_huggingface_hub_for_transformers,
    check_python_version,
    main,
)


def _constraint(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "serve.txt"
    path.write_text(dedent(body).strip() + "\n", encoding="utf-8")
    return path


def test_check_python_version_pass() -> None:
    with patch("sys.version_info", new=(3, 12, 13, "final", 0)):
        result = check_python_version()
    assert result.ok
    assert "fix" not in (result.fix or "")


def test_check_python_version_fail_records_fix() -> None:
    with patch("sys.version_info", new=(3, 11, 9, "final", 0)):
        result = check_python_version()
    assert not result.ok
    assert result.fix and "uv python install 3.12" in result.fix


def test_constraint_file_missing_package(tmp_path: Path) -> None:
    path = _constraint(tmp_path, "vllm>=0.11.0,<0.12")
    with patch("scripts.doctor._installed_version", return_value=None):
        checks = check_constraint_file(path, "serve")
    assert len(checks) == 1
    assert not checks[0].ok
    assert checks[0].detail == "not installed"
    assert checks[0].fix and "uv pip install" in checks[0].fix


def test_constraint_file_version_mismatch(tmp_path: Path) -> None:
    path = _constraint(tmp_path, "transformers>=4.45,<5.0")
    with patch("scripts.doctor._installed_version", return_value=Version("5.8.1")):
        checks = check_constraint_file(path, "serve")
    assert len(checks) == 1
    assert not checks[0].ok
    assert "5.8.1" in checks[0].detail
    assert "does not satisfy" in checks[0].detail


def test_constraint_file_satisfied(tmp_path: Path) -> None:
    path = _constraint(tmp_path, "transformers>=4.45,<5.0")
    with patch("scripts.doctor._installed_version", return_value=Version("4.57.6")):
        checks = check_constraint_file(path, "serve")
    assert checks[0].ok


def test_constraint_file_handles_comments_and_blanks(tmp_path: Path) -> None:
    path = _constraint(
        tmp_path,
        """
        # heading
        vllm>=0.11.0,<0.12

        transformers>=4.45,<5.0  # inline
        """,
    )

    def fake(name: str) -> Version:
        return Version({"vllm": "0.11.0", "transformers": "4.57.6"}[name])

    with patch("scripts.doctor._installed_version", side_effect=fake):
        checks = check_constraint_file(path, "serve")

    names = {c.name for c in checks}
    assert names == {"serve: vllm", "serve: transformers"}
    assert all(c.ok for c in checks)


def test_constraint_file_missing_path_is_noop(tmp_path: Path) -> None:
    checks = check_constraint_file(tmp_path / "does-not-exist.txt", "serve")
    assert checks == []


def test_executable_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/bin/{name}")
    result = check_executable("vllm", "fix-cmd")
    assert result.ok
    assert result.detail == "/fake/bin/vllm"


def test_executable_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    result = check_executable("vllm", "fix-cmd")
    assert not result.ok
    assert result.fix == "fix-cmd"


def test_huggingface_hub_guard_pass() -> None:
    # transformers 4.x + huggingface-hub 0.x → OK.
    def fake(name: str) -> Version:
        return Version({"transformers": "4.57.6", "huggingface-hub": "0.36.2"}[name])

    with patch("scripts.doctor._installed_version", side_effect=fake):
        result = check_huggingface_hub_for_transformers()
    assert result.ok


def test_huggingface_hub_guard_fail() -> None:
    # transformers 4.x + huggingface-hub 1.x → broken combo.
    def fake(name: str) -> Version:
        return Version({"transformers": "4.57.6", "huggingface-hub": "1.16.1"}[name])

    with patch("scripts.doctor._installed_version", side_effect=fake):
        result = check_huggingface_hub_for_transformers()
    assert not result.ok
    assert "1.16.1" in result.detail
    assert result.fix and "constraints/serve.txt" in result.fix


def test_huggingface_hub_guard_skipped_when_transformers_5x() -> None:
    # transformers 5.x doesn't have the pinning bug we're guarding against.
    def fake(name: str) -> Version:
        return Version({"transformers": "5.8.1", "huggingface-hub": "1.16.1"}[name])

    with patch("scripts.doctor._installed_version", side_effect=fake):
        result = check_huggingface_hub_for_transformers()
    assert result.ok


def test_huggingface_hub_guard_handles_missing_transformers() -> None:
    with patch("scripts.doctor._installed_version", return_value=None):
        result = check_huggingface_hub_for_transformers()
    assert not result.ok
    assert "not installed" in result.detail


def test_check_emit_works(capsys: pytest.CaptureFixture[str]) -> None:
    Check(name="x", ok=True, detail="ok").emit()
    out = capsys.readouterr().out
    assert "x" in out
    assert "ok" in out

    Check(name="y", ok=False, detail="bad", fix="run this").emit()
    out = capsys.readouterr().out
    assert "y" in out
    assert "bad" in out
    assert "fix: run this" in out


def test_main_exits_nonzero_when_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.doctor.run_checks",
        lambda **_: [Check(name="x", ok=False, detail="d", fix=None)],
    )
    assert main([]) == 1


def test_main_exits_zero_when_all_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.doctor.run_checks",
        lambda **_: [Check(name="x", ok=True, detail="d")],
    )
    assert main([]) == 0


def test_main_no_eval_flag_skips_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_run(*, eval_too: bool) -> list[Check]:
        captured["eval_too"] = eval_too
        return [Check(name="x", ok=True, detail="d")]

    monkeypatch.setattr("scripts.doctor.run_checks", fake_run)
    main(["--no-eval"])
    assert captured == {"eval_too": False}
