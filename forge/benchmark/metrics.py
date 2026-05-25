"""Parser for ``vllm bench serve --save-result`` JSON output.

The schema is documented loosely upstream and shifts between vLLM versions, so
the parser is intentionally permissive: it pulls the keys we care about and
ignores everything else. Schema drift surfaces as ``KeyError`` in tests against
the golden fixture, which is what we want — silent drift is the failure mode
that ruins a paid run after the fact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LatencyDistribution:
    """A single latency metric expressed in milliseconds."""

    mean_ms: float
    median_ms: float
    p99_ms: float
    std_ms: float | None = None

    @classmethod
    def from_vllm(cls, payload: dict[str, object], prefix: str) -> LatencyDistribution:
        """Extract a vLLM-style ``{prefix}_ms`` triplet (mean / median / p99) from a payload."""
        mean = _require_float(payload, f"mean_{prefix}_ms")
        median = _require_float(payload, f"median_{prefix}_ms")
        p99 = _require_float(payload, f"p99_{prefix}_ms")
        std = _optional_float(payload, f"std_{prefix}_ms")
        return cls(mean_ms=mean, median_ms=median, p99_ms=p99, std_ms=std)


@dataclass(frozen=True)
class BenchmarkRow:
    """One benchmark run reduced to its load-bearing numbers.

    ``concurrency`` is read from the input config (vLLM echoes ``max_concurrency``
    in the result but using our value avoids a parse-time ambiguity).
    """

    model: str
    backend: str
    dataset: str
    num_prompts: int
    concurrency: int
    seed: int
    duration_seconds: float
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    ttft: LatencyDistribution
    tpot: LatencyDistribution
    itl: LatencyDistribution | None
    input_lens_mean: float | None
    output_lens_mean: float | None
    source_path: str = field(repr=False)

    def to_dict(self) -> dict[str, object]:
        """Plain-dict view suitable for ``json.dumps`` and ``pandas.DataFrame``."""
        return asdict(self)


def parse_result(path: Path, concurrency: int) -> BenchmarkRow:
    """Load one ``vllm bench serve`` JSON and reduce it to a ``BenchmarkRow``.

    ``concurrency`` is the value our config used. We pass it explicitly rather
    than reading from the JSON because vLLM omits the field when concurrency is
    unbounded.
    """
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected JSON object at top level, got {type(payload).__name__}")

    return BenchmarkRow(
        model=_require_str(payload, "model_id"),
        backend=_require_str(payload, "backend"),
        dataset=_optional_str(payload, "dataset_name") or "unknown",
        num_prompts=_require_int(payload, "num_prompts"),
        concurrency=concurrency,
        seed=_optional_int(payload, "seed") or 0,
        duration_seconds=_require_float(payload, "duration"),
        request_throughput=_require_float(payload, "request_throughput"),
        output_throughput=_require_float(payload, "output_throughput"),
        total_token_throughput=_require_float(payload, "total_token_throughput"),
        ttft=LatencyDistribution.from_vllm(payload, "ttft"),
        tpot=LatencyDistribution.from_vllm(payload, "tpot"),
        itl=_optional_latency(payload, "itl"),
        input_lens_mean=_mean_or_none(payload.get("input_lens")),
        output_lens_mean=_mean_or_none(payload.get("output_lens")),
        source_path=str(path),
    )


def load_results(result_dir: Path, concurrency_levels: list[int], name: str) -> list[BenchmarkRow]:
    """Load every result JSON for a sweep, in concurrency order.

    Missing files raise ``FileNotFoundError`` rather than silently producing a
    partial dataset — chart pipelines fed partial data make misleading plots.
    """
    rows: list[BenchmarkRow] = []
    for concurrency in concurrency_levels:
        path = result_dir / f"{name}-c{concurrency:04d}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing benchmark result: {path}")
        rows.append(parse_result(path, concurrency))
    return rows


def _require_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise KeyError(f"missing or non-string field: {key}")
    return value


def _optional_str(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _require_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise KeyError(f"missing or non-int field: {key}")
    return value


def _optional_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _require_float(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise KeyError(f"missing or non-numeric field: {key}")
    return float(value)


def _optional_float(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _optional_latency(payload: dict[str, object], prefix: str) -> LatencyDistribution | None:
    if f"mean_{prefix}_ms" not in payload:
        return None
    return LatencyDistribution.from_vllm(payload, prefix)


def _mean_or_none(value: object) -> float | None:
    if not isinstance(value, list) or not value:
        return None
    nums = [v for v in value if isinstance(v, int | float) and not isinstance(v, bool)]
    return sum(nums) / len(nums) if nums else None
