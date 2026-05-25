"""AWQ quantization recipe.

The primary benchmark uses a community pre-quantized AWQ checkpoint
(``hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4``). This module exists for
reproducibility: it documents and executes the same recipe so a reader can
verify the checkpoint or quantize a different base model with our exact params.

Running the quantization requires a CUDA GPU and ~24 GB of VRAM for Llama 3.1
8B. It does not run on M1 — the calibration pass needs GPU matmul. The CLI in
``scripts/quantize.py`` errors loudly if invoked on a CPU-only host.
"""

from forge.quantization.awq import AWQConfig, quantize

__all__ = ["AWQConfig", "quantize"]
