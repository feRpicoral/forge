"""Subprocess wrapper around ``lm-eval`` (lm-evaluation-harness CLI).

lm-eval supports ``--model local-completions`` which speaks the OpenAI
Completions API — perfect for evaluating a vLLM server's quality without
loading the model in lm-eval's own process. We use the chat completions
endpoint here since our serving target is instruct-tuned.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
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
    """
    lm_eval = lm_eval_executable or shutil.which("lm-eval") or shutil.which("lm_eval")
    if not lm_eval:
        raise FileNotFoundError(
            "`lm-eval` not found on PATH. Install with `uv pip install -c constraints/eval.txt lm-eval`."
        )

    suite.result_dir.mkdir(parents=True, exist_ok=True)

    cmd = _build_command(lm_eval, suite)
    print(f"[forge] eval suite={suite.name} variant={suite.variant}", file=sys.stderr)
    print(f"[forge] tasks: {','.join(suite.tasks)}", file=sys.stderr)
    print(f"[forge] exec: {' '.join(cmd)}", file=sys.stderr)

    proc = subprocess.run(cmd, check=False)
    return RunOutcome(result_path=suite.result_path, returncode=proc.returncode)


def _build_command(lm_eval: str, suite: EvalSuite) -> list[str]:
    # local-completions speaks the OpenAI Completions API. Use openai-chat-completions
    # for chat-tuned models served via /v1/chat/completions.
    model_args = (
        f"model={suite.model},"
        f"base_url={suite.base_url.rstrip('/')}/chat/completions,"
        "tokenized_requests=False,"
        "num_concurrent=4"
    )
    cmd: list[str] = [
        lm_eval,
        "--model",
        "local-chat-completions",
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
