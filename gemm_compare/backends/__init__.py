"""
Optional backends (import only what you need — each may require extra packages).

- ``gemm_compare.backends.nvfp`` — Cutlass vs nvfp_cpp_emul
- ``gemm_compare.backends.mxfp`` — flashinfer mm_fp4 vs dequant reference

Use ``build_gemm_compare_fns(backend, **kwargs)`` to obtain ``(quant_fn, real_fn, emul_fn, meta)``
with backend-specific keyword arguments filtered automatically.
"""
from __future__ import annotations

from typing import Any, Callable

_MXFP_KW = frozenset({"group_size", "out_dtype", "mm_backend"})
_NVFP_KW = frozenset(
    {"out_dtype", "m_chunk_size", "stage3_rounding", "stage4_rounding"}
)


def build_gemm_compare_fns(
    backend: str = "mxfp",
    **kwargs: Any,
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], dict[str, Any]]:
    """Return ``(quant_fn, real_fn, emul_fn, meta)`` for the given backend name.

    Unknown kwargs for a backend are ignored so callers can pass a merged option dict.
    """
    b = backend.lower()
    if b == "mxfp":
        from .mxfp import build_mxfp_fns

        kw = {k: v for k, v in kwargs.items() if k in _MXFP_KW}
        return build_mxfp_fns(**kw)
    if b == "nvfp":
        from .nvfp import build_nvfp_fns

        kw = {k: v for k, v in kwargs.items() if k in _NVFP_KW}
        return build_nvfp_fns(**kw)
    raise ValueError(f"unknown backend {backend!r}; expected 'mxfp' or 'nvfp'")


__all__ = ["build_gemm_compare_fns"]
