"""
MXFP4: ``flashinfer.gemm.mm_fp4`` (real) vs dequantized BF16 matmul (reference emulation).

Uses ``mxfp.dequant_mxfp4`` from ``mxfp_cpp_emul/mxfp.py`` (directory added to ``sys.path``).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MXFP_PKG_DIR = _REPO_ROOT / "mxfp_cpp_emul"
if str(_MXFP_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_MXFP_PKG_DIR))


@dataclass
class MXFPQuantState:
    a_fp4: torch.Tensor
    b_fp4: torch.Tensor
    a_scale: torch.Tensor
    b_scale: torch.Tensor
    group_size: int
    out_dtype: torch.dtype
    mm_backend: str


def build_mxfp_fns(
    *,
    group_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
    mm_backend: str = "cudnn",
):
    import flashinfer
    from mxfp import dequant_mxfp4

    def quant_fn(a: torch.Tensor, b: torch.Tensor) -> MXFPQuantState:
        a_fp4, a_scale = flashinfer.mxfp4_quantize(a)
        b_fp4, b_scale = flashinfer.mxfp4_quantize(b)
        return MXFPQuantState(
            a_fp4=a_fp4,
            b_fp4=b_fp4,
            a_scale=a_scale,
            b_scale=b_scale,
            group_size=group_size,
            out_dtype=out_dtype,
            mm_backend=mm_backend,
        )

    def real_fn(state: MXFPQuantState) -> torch.Tensor:
        import flashinfer

        return flashinfer.gemm.mm_fp4(
            a=state.a_fp4,
            b=state.b_fp4.T,
            a_descale=state.a_scale,
            b_descale=state.b_scale.T,
            out_dtype=state.out_dtype,
            block_size=state.group_size,
            use_nvfp4=False,
            backend=state.mm_backend,
        )

    def emul_fn(state: MXFPQuantState) -> torch.Tensor:
        a_deq = dequant_mxfp4(
            state.a_fp4,
            state.a_scale,
            group_size=state.group_size,
            out_dtype=state.out_dtype,
        )
        b_deq = dequant_mxfp4(
            state.b_fp4,
            state.b_scale,
            group_size=state.group_size,
            out_dtype=state.out_dtype,
        )
        return a_deq @ b_deq.T

    meta: dict[str, Any] = {
        "backend": "mxfp",
        "group_size": group_size,
        "out_dtype": str(out_dtype),
        "mm_backend": mm_backend,
    }
    return quant_fn, real_fn, emul_fn, meta
