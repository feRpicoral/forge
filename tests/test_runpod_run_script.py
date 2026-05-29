from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_runpod_script_fails_gated_model_without_hf_auth(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env.pop("HUGGING_FACE_HUB_TOKEN", None)
    env["HOME"] = str(tmp_path / "home")
    env["HF_HOME"] = str(tmp_path / "hf")

    result = subprocess.run(
        ["bash", "deploy/runpod-run.sh", "--variant", "bf16", "--only-eval"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "missing Hugging Face auth" in result.stderr


def test_runpod_script_rejects_skipping_bench_and_eval() -> None:
    result = subprocess.run(
        ["bash", "deploy/runpod-run.sh", "--variant", "awq", "--skip-bench", "--skip-eval"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "nothing to run" in result.stderr
