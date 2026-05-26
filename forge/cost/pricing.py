"""Reference pricing tables for the cost comparison.

GPU hourly rates and commercial-API per-token prices change frequently. The
values here are pinned at a point in time and recorded in DECISIONS.md; before
the paid benchmark, refresh both tables against current rates and re-run the
chart pipeline. Every cost-comparison plot in the README must cite the date
these numbers were collected.

Sources:
- RunPod community tier pricing: https://runpod.io/pricing (as of 2026-05-26)
- OpenAI API pricing:            https://openai.com/api/pricing (as of 2026-05-26)
- Anthropic Claude API pricing:  https://www.anthropic.com/pricing (as of 2026-05-26)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuTier:
    name: str
    hourly_usd: float
    vram_gb: int
    notes: str


@dataclass(frozen=True)
class ApiPricing:
    """Per-1M-token prices for a commercial API."""

    name: str
    input_usd_per_1m: float
    output_usd_per_1m: float

    def blended_per_1m(self, input_share: float = 0.5) -> float:
        """Single-number cost assuming a given input/output mix.

        ``input_share`` is the fraction of tokens that are prompt vs.
        completion. Defaults to 50/50, matching the ShareGPT trace's rough mix.
        """
        if not 0.0 <= input_share <= 1.0:
            raise ValueError(f"input_share must be in [0, 1], got {input_share}")
        output_share = 1.0 - input_share
        return self.input_usd_per_1m * input_share + self.output_usd_per_1m * output_share


# Single source of truth for GPU pricing. Add tiers, don't redefine them inline.
GPU_TIERS: dict[str, GpuTier] = {
    "runpod-rtx-4090-community": GpuTier(
        name="RunPod RTX 4090 (Community)",
        hourly_usd=0.34,
        vram_gb=24,
        notes="The Forge benchmark target. Fits Llama 3.1 8B BF16 + KV cache headroom.",
    ),
    "runpod-a100-40gb-community": GpuTier(
        name="RunPod A100 40GB (Community)",
        hourly_usd=1.19,
        vram_gb=40,
        notes="Out of budget for Forge; included for sensitivity analysis.",
    ),
    "runpod-a100-80gb-community": GpuTier(
        name="RunPod A100 80GB (Community)",
        hourly_usd=1.89,
        vram_gb=80,
        notes="Out of budget for Forge; included for sensitivity analysis.",
    ),
    "runpod-h100-80gb-community": GpuTier(
        name="RunPod H100 80GB (Community)",
        hourly_usd=2.99,
        vram_gb=80,
        notes="Native FP8. Out of budget for Forge; reference for cost-scaling discussion.",
    ),
}


# Approximate commercial API pricing as of 2026-05-26. Update the numbers AND
# the source links above when refreshing.
API_PRICING: dict[str, ApiPricing] = {
    "gpt-4o": ApiPricing(name="GPT-4o", input_usd_per_1m=2.50, output_usd_per_1m=10.00),
    "gpt-4o-mini": ApiPricing(name="GPT-4o mini", input_usd_per_1m=0.15, output_usd_per_1m=0.60),
    "claude-sonnet-4-6": ApiPricing(
        name="Claude Sonnet 4.6", input_usd_per_1m=3.00, output_usd_per_1m=15.00
    ),
    "claude-haiku-4-5": ApiPricing(
        name="Claude Haiku 4.5", input_usd_per_1m=0.80, output_usd_per_1m=4.00
    ),
}
