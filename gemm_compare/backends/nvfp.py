"""
NVFP4 scaled GEMM: Cutlass / nvfp.ops vs C++ ``emulated_scaled_fp4_mm`` (nvfp_cpp_emul).

Requires ``nvfp`` (ops + pseudo_quant) and CUDA. JIT-builds the extension on first use.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NVFP_CPP_PARENT = _REPO_ROOT / "nvfp_cpp_emul"
if _NVFP_CPP_PARENT.is_dir() and str(_NVFP_CPP_PARENT) not in sys.path:
    sys.path.insert(0, str(_NVFP_CPP_PARENT))

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0


def global_scale_nvfp(t: torch.Tensor) -> torch.Tensor:
    amax = torch.abs(t).max().to(torch.float32).item()
    g = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (amax if amax > 0 else 1.0)
    return torch.tensor([g], device=t.device, dtype=torch.float32)


@dataclass
class NVFPQuantState:
    a_fp4: torch.Tensor
    b_fp4: torch.Tensor
    scale_a: torch.Tensor
    scale_b: torch.Tensor
    alpha: torch.Tensor
    out_dtype: torch.dtype
    w_reduce: int
    group_size: int
    m_chunk_size: int
    stage3_rounding: int
    stage4_rounding: int


def _dtype_to_str(dt: torch.dtype) -> str:
    s = str(dt)
    return s.removeprefix("torch.")


def _dtype_from_str(s: str) -> torch.dtype:
    name = s.removeprefix("torch.")
    return getattr(torch, name)


def nvfp_quant_state_to_cpu_dict(state: NVFPQuantState) -> dict[str, Any]:
    """Portable dict (tensors on CPU) for cross-machine rollout artifacts."""
    return {
        "kind": "nvfp",
        "a_fp4": state.a_fp4.detach().cpu().contiguous(),
        "b_fp4": state.b_fp4.detach().cpu().contiguous(),
        "scale_a": state.scale_a.detach().cpu().contiguous(),
        "scale_b": state.scale_b.detach().cpu().contiguous(),
        "alpha": state.alpha.detach().cpu().contiguous(),
        "out_dtype": _dtype_to_str(state.out_dtype),
        "w_reduce": int(state.w_reduce),
        "group_size": int(state.group_size),
        "m_chunk_size": int(state.m_chunk_size),
        "stage3_rounding": int(state.stage3_rounding),
        "stage4_rounding": int(state.stage4_rounding),
    }


def nvfp_quant_state_from_dict(d: dict[str, Any], *, device: str | torch.device) -> NVFPQuantState:
    if d.get("kind") != "nvfp":
        raise ValueError(f"expected NVFP state dict, got kind={d.get('kind')!r}")
    return NVFPQuantState(
        a_fp4=d["a_fp4"].to(device),
        b_fp4=d["b_fp4"].to(device),
        scale_a=d["scale_a"].to(device),
        scale_b=d["scale_b"].to(device),
        alpha=d["alpha"].to(device),
        out_dtype=_dtype_from_str(d["out_dtype"]),
        w_reduce=int(d["w_reduce"]),
        group_size=int(d["group_size"]),
        m_chunk_size=int(d["m_chunk_size"]),
        stage3_rounding=int(d["stage3_rounding"]),
        stage4_rounding=int(d["stage4_rounding"]),
    )


def build_nvfp_emul_fn(
    *,
    out_dtype: torch.dtype = torch.float16,
    m_chunk_size: int = 128,
    stage3_rounding: int | None = None,
    stage4_rounding: int | None = None,
):
    """C++ emulation only (no ``nvfp.ops`` / Cutlass). For import-side replay."""
    from nvfp_cpp_emul import RZ, emulated_scaled_fp4_mm

    if stage3_rounding is None:
        stage3_rounding = RZ
    if stage4_rounding is None:
        stage4_rounding = RZ

    def emul_fn(state: NVFPQuantState) -> torch.Tensor:
        return emulated_scaled_fp4_mm(
            state.a_fp4,
            state.b_fp4,
            state.scale_a,
            state.scale_b,
            state.alpha,
            m_chunk_size=state.m_chunk_size,
            stage3_rounding=state.stage3_rounding,
            stage4_rounding=state.stage4_rounding,
        )

    meta: dict[str, Any] = {
        "backend": "nvfp",
        "out_dtype": str(out_dtype),
        "emulation_only": True,
    }
    return emul_fn, meta


def build_nvfp_fns(
    *,
    out_dtype: torch.dtype = torch.float16,
    m_chunk_size: int = 128,
    stage3_rounding: int | None = None,
    stage4_rounding: int | None = None,
):
    import nvfp.ops as ops
    from nvfp_cpp_emul import RZ

    if stage3_rounding is None:
        stage3_rounding = RZ
    if stage4_rounding is None:
        stage4_rounding = RZ

    emul_fn, _ = build_nvfp_emul_fn(
        out_dtype=out_dtype,
        m_chunk_size=m_chunk_size,
        stage3_rounding=stage3_rounding,
        stage4_rounding=stage4_rounding,
    )

    def quant_fn(a: torch.Tensor, b: torch.Tensor) -> NVFPQuantState:
        a_gs = global_scale_nvfp(a)
        b_gs = global_scale_nvfp(b)
        alpha_val = 1.0 / max(a_gs.item() * b_gs.item(), 1e-8)
        alpha = torch.tensor([alpha_val], device=a.device, dtype=torch.float32)
        a_fp4, scale_a = ops.scaled_fp4_quant(a, a_gs)
        b_fp4, scale_b = ops.scaled_fp4_quant(b, b_gs)
        a_fp4 = a_fp4.contiguous().view(torch.uint8)
        b_fp4 = b_fp4.contiguous().view(torch.uint8)
        return NVFPQuantState(
            a_fp4=a_fp4,
            b_fp4=b_fp4,
            scale_a=scale_a,
            scale_b=scale_b,
            alpha=alpha,
            out_dtype=out_dtype,
            w_reduce=4,
            group_size=16,
            m_chunk_size=m_chunk_size,
            stage3_rounding=stage3_rounding,
            stage4_rounding=stage4_rounding,
        )

    def real_fn(state: NVFPQuantState) -> torch.Tensor:
        return ops.cutlass_scaled_fp4_mm(
            state.a_fp4,
            state.b_fp4,
            state.scale_a,
            state.scale_b,
            state.alpha,
            state.out_dtype,
        )

    meta: dict[str, Any] = {
        "backend": "nvfp",
        "out_dtype": str(out_dtype),
    }
    return quant_fn, real_fn, emul_fn, meta
