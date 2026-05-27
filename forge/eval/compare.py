"""Parse lm-evaluation-harness output and compute full-vs-quantized deltas.

lm-eval writes a JSON with a ``results`` mapping (task → metric → value). We
extract the canonical metric per task (``acc_norm`` for HellaSwag, ``acc`` for
MMLU, ``exact_match,strict-match`` for GSM8K) and produce a ``QualityDelta``
summarizing the percentage-point retention against the baseline.

The aggregate retention is reported as a simple mean across tasks. A weighted
aggregate (by sample count or task importance) is a future improvement — the
simple mean keeps the methodology trivially explainable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class _MetricSpec:
    """Canonical metric for a task, plus the ordered keys to look for in lm-eval JSON.

    lm-eval emits metric keys as ``<metric>,<filter>`` (e.g. ``acc,none``,
    ``exact_match,strict-match``). ``accepted_keys`` lists every key we accept,
    in preference order, so a dict-iteration-order flip in lm-eval's output
    can't silently switch which score lands in the retention chart.
    """

    name: str
    accepted_keys: tuple[str, ...]


# Canonical metric per task. GSM8K's ``strict-match`` is preferred over
# ``flexible-extract`` because the strict filter is the conservative number we
# cite in the methodology section.
_PRIMARY_METRIC: dict[str, _MetricSpec] = {
    "mmlu": _MetricSpec(name="acc", accepted_keys=("acc,none",)),
    "gsm8k": _MetricSpec(
        name="exact_match",
        accepted_keys=("exact_match,strict-match", "exact_match,flexible-extract"),
    ),
    "hellaswag": _MetricSpec(name="acc_norm", accepted_keys=("acc_norm,none",)),
}


@dataclass(frozen=True)
class TaskResult:
    task: str
    metric: str
    score: float
    samples: int | None = None


@dataclass(frozen=True)
class TaskDelta:
    task: str
    metric: str
    baseline_score: float
    candidate_score: float
    absolute_delta_pp: float  # candidate - baseline, percentage points
    retention_pct: float  # candidate / baseline as a percentage


@dataclass(frozen=True)
class QualityDelta:
    """Pairwise retention summary between a baseline and a candidate variant."""

    baseline_label: str
    candidate_label: str
    per_task: list[TaskDelta]
    mean_retention_pct: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_label": self.baseline_label,
            "candidate_label": self.candidate_label,
            "mean_retention_pct": self.mean_retention_pct,
            "per_task": [
                {
                    "task": d.task,
                    "metric": d.metric,
                    "baseline_score": d.baseline_score,
                    "candidate_score": d.candidate_score,
                    "absolute_delta_pp": d.absolute_delta_pp,
                    "retention_pct": d.retention_pct,
                }
                for d in self.per_task
            ],
            "notes": self.notes,
        }


def parse_lm_eval_output(path: Path) -> list[TaskResult]:
    """Reduce an lm-eval JSON to one ``TaskResult`` per task in ``_PRIMARY_METRIC``.

    Unknown tasks (not in ``_PRIMARY_METRIC``) are skipped — that keeps the
    comparison surface narrow and reproducible. Add tasks here, not in the
    config YAML alone.
    """
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected JSON object, got {type(payload).__name__}")

    results = payload.get("results")
    if not isinstance(results, dict):
        raise KeyError(f"{path}: missing 'results' object")

    out: list[TaskResult] = []
    for task, spec in _PRIMARY_METRIC.items():
        if task not in results:
            continue
        task_data = results[task]
        if not isinstance(task_data, dict):
            continue
        score = _extract_score(task_data, spec)
        if score is None:
            continue
        out.append(
            TaskResult(
                task=task,
                metric=spec.name,
                score=score,
                samples=_extract_samples(task_data),
            )
        )
    return out


def compute_deltas(
    baseline: list[TaskResult],
    candidate: list[TaskResult],
    *,
    baseline_label: str = "bf16",
    candidate_label: str = "awq",
) -> QualityDelta:
    """Compute per-task and aggregate retention of the candidate vs. the baseline.

    Tasks present in only one side are skipped and recorded in ``notes``.
    Retention is candidate/baseline as a percent. A retention of 100% means
    the candidate matches the baseline exactly.
    """
    baseline_by_task = {r.task: r for r in baseline}
    candidate_by_task = {r.task: r for r in candidate}
    common = sorted(set(baseline_by_task) & set(candidate_by_task))
    skipped = sorted(set(baseline_by_task) ^ set(candidate_by_task))

    per_task: list[TaskDelta] = []
    retentions: list[float] = []
    for task in common:
        b = baseline_by_task[task]
        c = candidate_by_task[task]
        if b.score == 0:
            continue
        retention = (c.score / b.score) * 100.0
        per_task.append(
            TaskDelta(
                task=task,
                metric=b.metric,
                baseline_score=b.score,
                candidate_score=c.score,
                absolute_delta_pp=(c.score - b.score) * 100.0,
                retention_pct=retention,
            )
        )
        retentions.append(retention)

    mean_retention = sum(retentions) / len(retentions) if retentions else 0.0
    notes = [f"skipped (not in both results): {t}" for t in skipped]
    return QualityDelta(
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        per_task=per_task,
        mean_retention_pct=mean_retention,
        notes=notes,
    )


def _extract_score(task_data: dict[str, object], spec: _MetricSpec) -> float | None:
    """Return the first accepted-key score, walking ``spec.accepted_keys`` in order.

    Iterating the spec's ordered keys (rather than the JSON's key order) is the
    whole point: lm-eval can emit several keys for the same metric (e.g.
    GSM8K's ``strict-match`` and ``flexible-extract``), and we want a
    deterministic choice regardless of dict ordering.
    """
    for key in spec.accepted_keys:
        value = task_data.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def _extract_samples(task_data: dict[str, object]) -> int | None:
    samples = task_data.get("samples")
    if isinstance(samples, int) and not isinstance(samples, bool):
        return samples
    return None
