"""YAML-driven benchmark sweep configuration.

A sweep describes a single server configuration (one model, one quantization)
benchmarked across a list of concurrency levels. Cross-quantization comparison
is orchestrated outside the harness by ``deploy/runpod-run.sh`` — it stops the
server, switches env vars, and re-runs the harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    num_prompts: int
    extra_args: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchSweep:
    """Declarative description of a benchmark run.

    Loaded from ``configs/bench-*.yaml``. Every concurrency level produces one
    result JSON in ``result_dir``.
    """

    name: str
    description: str
    model: str
    base_url: str
    backend: str
    dataset: DatasetConfig
    seed: int
    result_dir: Path
    concurrency_levels: list[int]

    def result_filename(self, concurrency: int) -> str:
        """File name for the result JSON at a given concurrency level."""
        return f"{self.name}-c{concurrency:04d}.json"

    def result_path(self, concurrency: int) -> Path:
        return self.result_dir / self.result_filename(concurrency)


def load_sweep(path: Path) -> BenchSweep:
    """Parse a benchmark sweep YAML.

    Raises ``KeyError`` if a required key is missing, ``TypeError`` if a value
    has the wrong type. Strict by design — sweep configs are the input to a
    paid GPU run, ambiguity is the enemy.
    """
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise TypeError(f"Expected a YAML mapping at {path}, got {type(raw).__name__}")

    dataset_raw = _require_mapping(raw, "dataset", path)
    extra_args_raw = dataset_raw.get("extra_args") or {}
    if not isinstance(extra_args_raw, dict):
        raise TypeError(
            f"{path}:dataset.extra_args must be a mapping, got {type(extra_args_raw).__name__}"
        )
    dataset = DatasetConfig(
        name=_require_str(dataset_raw, "name", f"{path}:dataset"),
        num_prompts=_require_int(dataset_raw, "num_prompts", f"{path}:dataset"),
        extra_args={str(k): str(v) for k, v in extra_args_raw.items()},
    )

    concurrency_raw = _require_list(raw, "concurrency_levels", path)
    if not concurrency_raw:
        raise ValueError(f"{path}: concurrency_levels must not be empty")
    concurrency_levels: list[int] = []
    for idx, value in enumerate(concurrency_raw):
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                f"{path}: concurrency_levels[{idx}] must be a positive int, got {value!r}"
            )
        concurrency_levels.append(value)

    return BenchSweep(
        name=_require_str(raw, "name", path),
        description=_require_str(raw, "description", path),
        model=_require_str(raw, "model", path),
        base_url=_require_str(raw, "base_url", path),
        backend=_require_str(raw, "backend", path),
        dataset=dataset,
        seed=_require_int(raw, "seed", path),
        result_dir=Path(_require_str(raw, "result_dir", path)),
        concurrency_levels=concurrency_levels,
    )


def _require_mapping(d: dict[str, object], key: str, ctx: object) -> dict[str, object]:
    value = d.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{ctx}: '{key}' must be a mapping, got {type(value).__name__}")
    return value


def _require_list(d: dict[str, object], key: str, ctx: object) -> list[object]:
    value = d.get(key)
    if not isinstance(value, list):
        raise TypeError(f"{ctx}: '{key}' must be a list, got {type(value).__name__}")
    return value


def _require_str(d: dict[str, object], key: str, ctx: object) -> str:
    value = d.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{ctx}: '{key}' must be a non-empty string, got {value!r}")
    return value


def _require_int(d: dict[str, object], key: str, ctx: object) -> int:
    value = d.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{ctx}: '{key}' must be an int, got {type(value).__name__}")
    return value
