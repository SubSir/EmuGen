#!/usr/bin/env python3
"""
Minimal example: three callables, no CUDA / flashinfer / nvfp.

Run from repo root::

  PYTHONPATH=. python gemm_compare/examples/toy_cpu.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from gemm_compare.runner import run_suite


def main() -> int:
    def quant_fn(a: torch.Tensor, b: torch.Tensor):
        return {"a": a, "b": b}

    def real_fn(state):
        return state["a"] @ state["b"].T

    def emul_fn(state):
        # Deliberately identical here; swap in your modeled kernel.
        return state["a"] @ state["b"].T

    return run_suite(
        quant_fn,
        real_fn,
        emul_fn,
        name="Toy FP32 matmul (CPU)",
        num_iterations=5,
        device="cpu",
        dtype=torch.float32,
        dims_m=[8],
        dims_n=[8],
        dims_k=[16],
        distributions=["normal"],
        seed=0,
    )


if __name__ == "__main__":
    sys.exit(main())
