"""
Compare hardware (or library) GEMM against an emulation or reference path.

Pass three callables: ``quant_fn(a, b) -> state``, ``real_fn(state)``, ``emul_fn(state)``.
"""

from .data import DataGenerator, DataGenerator_Abs
from .runner import compare_tensors, run_test_case, run_suite

__all__ = [
    "DataGenerator",
    "DataGenerator_Abs",
    "compare_tensors",
    "run_test_case",
    "run_suite",
]
