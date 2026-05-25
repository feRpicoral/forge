"""Subprocess wrapper around ``lm-eval`` (lm-evaluation-harness CLI).

lm-eval supports ``--model local-completions`` which speaks the OpenAI
Completions API — perfect for evaluating a vLLM server's quality without
loading the model in lm-eval's own process. We hit the raw ``/v1/completions``
endpoint because loglikelihood scoring (MMLU, HellaSwag) is unsupported via
chat completions.

lm-eval writes its output as ``<prefix>_<iso8601>.json`` inside the given
``--output_path`` directory — the suffix is non-deterministic. After the run
the wrapper discovers the produced file and (optionally) renames it to the
canonical ``<name>-<variant>.json`` form the rest of the pipeline expects.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from forge.eval.config import EvalSuite


@dataclass(frozen=True)
class RunOutcome:
    result_path: Path
    returncode: int

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


def run_suite(suite: EvalSuite, *, lm_eval_executable: str | None = None) -> RunOutcome:
    """Run the full task list against the configured server.

    Caller starts and stops the vLLM server. lm-eval CLI options are kept to
    the subset we actually use — adding more is a YAML schema change, not a
    runner change.

    After lm-eval completes, the timestamped file it produced is renamed to the
    canonical path the rest of the pipeline expects.
    """
    lm_eval = lm_eval_executable or shutil.which("lm-eval") or shutil.which("lm_eval")
    if not lm_eval:
        raise FileNotFoundError(
            "`lm-eval` not found on PATH. Install with "
            '`uv pip install -c constraints/eval.txt "lm-eval[api]"`.'
        )

    suite.result_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = time.time()

    cmd = _build_command(lm_eval, suite)
    print(f"[forge] eval suite={suite.name} variant={suite.variant}", file=sys.stderr)
    print(f"[forge] tasks: {','.join(suite.tasks)}", file=sys.stderr)
    print(f"[forge] exec: {' '.join(cmd)}", file=sys.stderr)

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        return RunOutcome(result_path=suite.result_path, returncode=proc.returncode)

    canonical = _canonicalize_output(suite, run_started_at)
    return RunOutcome(result_path=canonical, returncode=0)


def _canonicalize_output(suite: EvalSuite, run_started_at: float) -> Path:
    """Find the timestamped JSON lm-eval just wrote and rename it canonically.

    Newest JSON in ``suite.result_dir`` created after ``run_started_at`` wins.
    If a canonical file already exists, it's overwritten (each variant has one
    authoritative result file at a time).
    """
    candidates = sorted(
        (p for p in suite.result_dir.glob("*.json") if p.stat().st_mtime >= run_started_at - 1),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print(
            f"[forge] WARNING: lm-eval reported success but no JSON found in {suite.result_dir}",
            file=sys.stderr,
        )
        return suite.result_path
    latest = candidates[0]
    if latest == suite.result_path:
        return latest
    suite.result_path.unlink(missing_ok=True)
    latest.rename(suite.result_path)
    return suite.result_path


def _build_command(lm_eval: str, suite: EvalSuite) -> list[str]:
    # local-completions speaks the raw /v1/completions API. MMLU/HellaSwag use
    # loglikelihood scoring, which the chat-completions backend doesn't support;
    # local-completions handles both loglikelihood and generate_until tasks.
    model_args = (
        f"model={suite.model},"
        f"base_url={suite.base_url.rstrip('/')}/completions,"
        "tokenized_requests=False,"
        "num_concurrent=4"
    )
    cmd: list[str] = [
        lm_eval,
        "--model",
        "local-completions",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(suite.tasks),
        "--num_fewshot",
        str(suite.num_fewshot),
        "--batch_size",
        str(suite.batch_size),
        "--seed",
        str(suite.seed),
        "--output_path",
        str(suite.result_path),
    ]
    if suite.limit is not None:
        cmd.extend(["--limit", str(suite.limit)])
    return cmd
