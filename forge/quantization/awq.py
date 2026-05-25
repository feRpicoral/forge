"""AWQ-INT4 quantization recipe via AutoAWQ.

The defaults match the recipe used by ``casperhansen/autoawq`` and the
``hugging-quants`` community checkpoints: 4-bit weights, group size 128, zero-
point, GEMM kernels. These are the same params our serving layer dispatches
through vLLM's Marlin kernels.

``autoawq`` and ``torch`` are imported lazily inside ``quantize`` so this
module can be imported on machines without a GPU runtime (CI, M1) — useful
for tests and dry-run inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AWQConfig:
    """AWQ recipe parameters.

    Defaults match the canonical 4-bit recipe shipped with AutoAWQ:
    - ``w_bit=4`` — 4-bit weights.
    - ``q_group_size=128`` — group quantization, 128 weights share one scale.
    - ``zero_point=True`` — asymmetric quantization (per-group zero offset).
    - ``version="GEMM"`` — kernel family. ``GEMV`` is the alternative;
      ``GEMM`` is faster on modern GPUs and what Marlin accelerates.
    """

    source_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_path: Path = field(default_factory=lambda: Path("./out/llama-3.1-8b-awq-int4"))
    w_bit: int = 4
    q_group_size: int = 128
    zero_point: bool = True
    version: str = "GEMM"

    @property
    def quant_config(self) -> dict[str, object]:
        """Mapping accepted by ``AutoAWQForCausalLM.quantize(quant_config=…)``."""
        return {
            "w_bit": self.w_bit,
            "q_group_size": self.q_group_size,
            "zero_point": self.zero_point,
            "version": self.version,
        }

    def describe(self) -> str:
        """Human-readable summary, for stderr logging at quantize-time."""
        return (
            f"AWQ-INT4 recipe: w_bit={self.w_bit}, q_group_size={self.q_group_size}, "
            f"zero_point={self.zero_point}, version={self.version}\n"
            f"  source: {self.source_model}\n"
            f"  output: {self.output_path}"
        )


def quantize(config: AWQConfig) -> Path:
    """Run AWQ quantization and write the result to ``config.output_path``.

    Requires a CUDA GPU and ``autoawq`` installed (``uv pip install autoawq``).
    Returns the absolute output path on success. Calibration uses AutoAWQ's
    default 512-sample slice of ``mit-han-lab/pile-val-backup``.
    """
    # Lazy imports — autoawq pulls in torch + CUDA-coupled deps and is only
    # available where quantization can actually run. Keep the module importable
    # on M1 for tests + dry-run. Both modules are configured as
    # ``ignore_missing_imports`` in mypy so the lazy import works in CI too.
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    output_path = config.output_path.absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    # transformers ships no .pyi stubs. The ``unused-ignore`` suffix keeps the
    # comment harmless on CI (where transformers isn't installed and the call
    # is Any-typed anyway).
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call, unused-ignore]
        config.source_model, trust_remote_code=True
    )
    model = AutoAWQForCausalLM.from_pretrained(
        config.source_model,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    model.quantize(tokenizer, quant_config=config.quant_config)
    model.save_quantized(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    return output_path
