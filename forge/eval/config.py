"""YAML-driven quality eval suite configuration.

A suite describes a single model variant being evaluated across a list of
lm-evaluation-harness tasks. Cross-variant comparison (BF16 vs quantized) is
orchestrated outside the suite: run the suite twice with different ``variant``
labels, then feed both outputs to ``compute_deltas``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EvalSuite:
    name: str
    description: str
    variant: str  # e.g. "bf16" or "awq" — used in output filename
    model: str
    base_url: str
    tasks: list[str]
    num_fewshot: int
    batch_size: int
    limit: int | None  # cap samples per task (smoke); None = full task
    seed: int
    result_dir: Path

    @property
    def result_path(self) -> Path:
        return self.result_dir / f"{self.name}-{self.variant}.json"


def load_suite(path: Path) -> EvalSuite:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a YAML mapping at {path}, got {type(raw).__name__}")

    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise TypeError(f"{path}: 'tasks' must be a non-empty list")
    tasks = [str(t) for t in tasks_raw]

    limit_raw = raw.get("limit")
    limit: int | None
    if limit_raw is None:
        limit = None
    elif isinstance(limit_raw, int) and not isinstance(limit_raw, bool) and limit_raw > 0:
        limit = limit_raw
    else:
        raise TypeError(f"{path}: 'limit' must be a positive int or null, got {limit_raw!r}")

    return EvalSuite(
        name=_require_str(raw, "name", path),
        description=_require_str(raw, "description", path),
        variant=_require_str(raw, "variant", path),
        model=_require_str(raw, "model", path),
        base_url=_require_str(raw, "base_url", path),
        tasks=tasks,
        num_fewshot=_require_int(raw, "num_fewshot", path),
        batch_size=_require_int(raw, "batch_size", path),
        limit=limit,
        seed=_require_int(raw, "seed", path),
        result_dir=Path(_require_str(raw, "result_dir", path)),
    )


def _require_str(d: dict[str, object], key: str, ctx: object) -> str:
    value = d.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{ctx}: '{key}' must be a non-empty string, got {value!r}")
    return value


def _require_int(d: dict[str, object], key: str, ctx: object) -> int:
    value = d.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{ctx}: '{key}' must be an int, got {value!r}")
    return value
