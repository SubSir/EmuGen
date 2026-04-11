"""
Python NVFP scaled FP4 GEMM emulation from a pre-built state (quantization is caller-owned).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .core import MMAEngine
from .rounding import RoundStrategy


@dataclass
class SearchNVFPEmulationState:
    """Quantized tensors + emulation knobs (W3, W4, W). Build this however you like upstream."""

    a_fp4: torch.Tensor
    b_fp4: torch.Tensor
    scale_a: torch.Tensor
    scale_b: torch.Tensor
    alpha: torch.Tensor
    out_dtype: torch.dtype
    w_stage3: int
    w_stage4: int
    w_reduce: int
    m_chunk_size: int
    # Integers match ``nvfp_cpp_emul.RZ`` / ``RNE``; :class:`RoundStrategy` also accepted.
    stage3_rounding: RoundStrategy | int
    stage4_rounding: RoundStrategy | int


def emulate_nvfp_scaled_fp4_mm(
    state: SearchNVFPEmulationState,
    *,
    unpack_fp4: Optional[Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]] = None,
    linearize_block_scales: Optional[Callable[[torch.Tensor, int, int], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Run ``MMAEngine.emulation_scaled_fp4_mm`` on ``state``.

    ``unpack_fp4(packed_uint8, shape)`` and ``linearize_block_scales(scale, rows, G)`` default to
    NVFP ``nvfp_unpack_fp4_to_fp16`` / ``nvfp_swizzled_block_scale_to_linear`` when omitted.
    """
    M = state.a_fp4.shape[0]
    N = state.b_fp4.shape[0]
    K = state.a_fp4.shape[1] * 2
    out = MMAEngine.emulation_scaled_fp4_mm(
        state.a_fp4,
        state.b_fp4,
        state.scale_a,
        state.scale_b,
        state.alpha,
        M,
        N,
        K,
        W_stage3=state.w_stage3,
        W_stage4=state.w_stage4,
        w_reduce=state.w_reduce,
        stage3_rounding=state.stage3_rounding,
        stage4_rounding=state.stage4_rounding,
        m_chunk_size=state.m_chunk_size,
        unpack_fp4=unpack_fp4,
        linearize_block_scales=linearize_block_scales,
    )
    assert state.out_dtype == torch.float16 # I'm lazy, you can change it to torch.bfloat16 if you want
    return out.to(state.out_dtype)
