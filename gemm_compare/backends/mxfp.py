"""
MXFP4: ``flashinfer.gemm.mm_fp4`` (real) vs Python ``MMAEngine`` via ``emulate_nvfp_scaled_fp4_mm``.

The ``mxfp_cpp_emul`` directory is on ``sys.path`` for ``mxfp`` helpers (unpack / scale linearize).

Bitwise ``real == emul`` is sensitive to ``out_dtype`` and to the dtype of ``A,B`` passed to
``mxfp4_quantize`` (``search/w3.py`` uses float16 for both; use the same in ``gemm_compare/cli.py``).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MXFP_PKG_DIR = _REPO_ROOT / "mxfp_cpp_emul"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_MXFP_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_MXFP_PKG_DIR))


@dataclass
class MXFPQuantState:
    a_fp4: torch.Tensor
    b_fp4: torch.Tensor
    a_scale: torch.Tensor
    b_scale: torch.Tensor
    alpha: torch.Tensor
    group_size: int
    out_dtype: torch.dtype
    mm_backend: str

    w_stage3: int
    w_stage4: int
    w_reduce: int
    m_chunk_size: int
    stage3_rounding: int
    stage4_rounding: int


def mxfp_state_as_nvfp_emulation(state: MXFPQuantState):
    """View MXFP tensors + ``alpha==1`` as :class:`search.emulation.gemm.SearchNVFPEmulationState` for shared MMA emu."""
    from search.emulation.gemm import SearchNVFPEmulationState

    return SearchNVFPEmulationState(
        a_fp4=state.a_fp4,
        b_fp4=state.b_fp4,
        scale_a=state.a_scale,
        scale_b=state.b_scale,
        alpha=state.alpha,
        out_dtype=state.out_dtype,
        w_stage3=state.w_stage3,
        w_stage4=state.w_stage4,
        w_reduce=state.w_reduce,
        group_size=state.group_size,
        m_chunk_size=state.m_chunk_size,
        stage3_rounding=state.stage3_rounding,
        stage4_rounding=state.stage4_rounding,
    )


def build_mxfp_fns(
    *,
    group_size: int = 32,
    out_dtype: torch.dtype = torch.float16,
    mm_backend: str = "cudnn",
    w_stage3: int = 25,
    w_stage4: int = 25,
    m_chunk_size: int = 128,
    stage3_rounding: int | None = None,
    stage4_rounding: int | None = None,
):
    from mxfp_cpp_emul import RZ

    if stage3_rounding is None:
        stage3_rounding = RZ
    if stage4_rounding is None:
        stage4_rounding = RZ

    from mxfp import mxfp_swizzled_scale_to_linear_fp32, unpack_mxfp4_to_fp16
    from search.emulation import emulate_nvfp_scaled_fp4_mm

    import flashinfer

    def quant_fn(a: torch.Tensor, b: torch.Tensor) -> MXFPQuantState:
        a_fp4, a_scale = flashinfer.mxfp4_quantize(a)
        b_fp4, b_scale = flashinfer.mxfp4_quantize(b)
        alpha = torch.tensor([1.0], device=a.device, dtype=torch.float32)
        return MXFPQuantState(
            a_fp4=a_fp4,
            b_fp4=b_fp4,
            a_scale=a_scale,
            b_scale=b_scale,
            alpha=alpha,
            group_size=group_size,
            out_dtype=out_dtype,
            mm_backend=mm_backend,
            w_stage3=w_stage3,
            w_stage4=w_stage4,
            w_reduce=2,
            m_chunk_size=m_chunk_size,
            stage3_rounding=stage3_rounding,
            stage4_rounding=stage4_rounding,
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
        return emulate_nvfp_scaled_fp4_mm(
            mxfp_state_as_nvfp_emulation(state),
            unpack_fp4=unpack_mxfp4_to_fp16,
            linearize_block_scales=mxfp_swizzled_scale_to_linear_fp32,
        )

    meta: dict[str, Any] = {
        "backend": "mxfp",
        "group_size": group_size,
        "out_dtype": str(out_dtype),
        "mm_backend": mm_backend,
    }
    return quant_fn, real_fn, emul_fn, meta
