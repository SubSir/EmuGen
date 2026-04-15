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


def _dtype_to_str(dt: torch.dtype) -> str:
    s = str(dt)
    return s.removeprefix("torch.")


def _dtype_from_str(s: str) -> torch.dtype:
    name = s.removeprefix("torch.")
    return getattr(torch, name)


def mxfp_quant_state_to_cpu_dict(state: MXFPQuantState) -> dict[str, Any]:
    """Portable dict (tensors on CPU) for cross-machine rollout artifacts."""
    return {
        "kind": "mxfp",
        "a_fp4": state.a_fp4.detach().cpu().contiguous(),
        "b_fp4": state.b_fp4.detach().cpu().contiguous(),
        "a_scale": state.a_scale.detach().cpu().contiguous(),
        "b_scale": state.b_scale.detach().cpu().contiguous(),
        "alpha": state.alpha.detach().cpu().contiguous(),
        "group_size": int(state.group_size),
        "out_dtype": _dtype_to_str(state.out_dtype),
        "mm_backend": str(state.mm_backend),
        "w_stage3": int(state.w_stage3),
        "w_stage4": int(state.w_stage4),
        "w_reduce": int(state.w_reduce),
        "m_chunk_size": int(state.m_chunk_size),
        "stage3_rounding": int(state.stage3_rounding),
        "stage4_rounding": int(state.stage4_rounding),
    }


def mxfp_quant_state_from_dict(d: dict[str, Any], *, device: str | torch.device) -> MXFPQuantState:
    if d.get("kind") != "mxfp":
        raise ValueError(f"expected MXFP state dict, got kind={d.get('kind')!r}")
    return MXFPQuantState(
        a_fp4=d["a_fp4"].to(device),
        b_fp4=d["b_fp4"].to(device),
        a_scale=d["a_scale"].to(device),
        b_scale=d["b_scale"].to(device),
        alpha=d["alpha"].to(device),
        group_size=int(d["group_size"]),
        out_dtype=_dtype_from_str(d["out_dtype"]),
        mm_backend=str(d["mm_backend"]),
        w_stage3=int(d["w_stage3"]),
        w_stage4=int(d["w_stage4"]),
        w_reduce=int(d["w_reduce"]),
        m_chunk_size=int(d["m_chunk_size"]),
        stage3_rounding=int(d["stage3_rounding"]),
        stage4_rounding=int(d["stage4_rounding"]),
    )


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


def build_mxfp_emul_fn(
    *,
    group_size: int = 32,
    out_dtype: torch.dtype = torch.float16,
    w_stage3: int = 25,
    w_stage4: int = 25,
    m_chunk_size: int = 128,
    stage3_rounding: int | None = None,
    stage4_rounding: int | None = None,
):
    """Python emulation path only (no flashinfer). For import-side replay."""
    from mxfp_cpp_emul import RZ

    if stage3_rounding is None:
        stage3_rounding = RZ
    if stage4_rounding is None:
        stage4_rounding = RZ

    from mxfp import mxfp_swizzled_scale_to_linear_fp32, unpack_mxfp4_to_fp16
    from search.emulation import emulate_nvfp_scaled_fp4_mm

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
        "emulation_only": True,
    }
    return emul_fn, meta


def build_mxfp_pseudo_fn():
    import flashinfer

    def pseudo_fn(state: MXFPQuantState) -> torch.Tensor:
        return flashinfer.mxfp4_dequantize(state.a_fp4, state.a_scale) @ flashinfer.mxfp4_dequantize(
            state.b_fp4, state.b_scale
        ).T

    meta: dict[str, Any] = {"backend": "mxfp", "pseudo_only": True}
    return pseudo_fn, meta


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
    return_pseudo: bool = False,
):
    from mxfp_cpp_emul import RZ

    if stage3_rounding is None:
        stage3_rounding = RZ
    if stage4_rounding is None:
        stage4_rounding = RZ

    emul_fn, _ = build_mxfp_emul_fn(
        group_size=group_size,
        out_dtype=out_dtype,
        w_stage3=w_stage3,
        w_stage4=w_stage4,
        m_chunk_size=m_chunk_size,
        stage3_rounding=stage3_rounding,
        stage4_rounding=stage4_rounding,
    )

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
    
    def pseudo_fn(state: MXFPQuantState) -> torch.Tensor:
        out = flashinfer.mxfp4_dequantize(state.a_fp4, state.a_scale) @ flashinfer.mxfp4_dequantize(state.b_fp4, state.b_scale).T
        return out.cuda()

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

    meta: dict[str, Any] = {
        "backend": "mxfp",
        "group_size": group_size,
        "out_dtype": str(out_dtype),
        "mm_backend": mm_backend,
    }
    if return_pseudo:
        return quant_fn, real_fn, emul_fn, pseudo_fn, meta
    return quant_fn, real_fn, emul_fn, meta
