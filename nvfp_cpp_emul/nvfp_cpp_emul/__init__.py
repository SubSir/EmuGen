"""
NVFP4 scaled GEMM emulation (libtorch C++), JIT-built via torch.utils.cpp_extension.load.

No pip install: the extension compiles on first use. Adjust JIT_EXTRA_CFLAGS (or call
configure_jit) before the first call to emulated_scaled_fp4_mm if you need custom flags.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

# Optional compile flags (empty by default; set before first emulated_scaled_fp4_mm()).
JIT_EXTRA_CFLAGS: list[str] = []
JIT_EXTRA_LDFLAGS: list[str] = []
JIT_EXTRA_INCLUDE_PATHS: list[str] = []

# Merged into load(); use configure_jit() so cache is cleared when changing options.
_LOAD_KWARGS: dict[str, Any] = {}


def configure_jit(
    *,
    extra_cflags: list[str] | None = None,
    extra_ldflags: list[str] | None = None,
    extra_include_paths: list[str] | None = None,
    **kwargs: Any,
) -> None:
    """Update arguments passed to torch.utils.cpp_extension.load; clears JIT cache."""
    global _LOAD_KWARGS
    if extra_cflags is not None:
        JIT_EXTRA_CFLAGS[:] = extra_cflags
    if extra_ldflags is not None:
        JIT_EXTRA_LDFLAGS[:] = extra_ldflags
    if extra_include_paths is not None:
        JIT_EXTRA_INCLUDE_PATHS[:] = extra_include_paths
    _LOAD_KWARGS.update(kwargs)
    _load_nvfp_emul.cache_clear()


@lru_cache(maxsize=1)
def _load_nvfp_emul():
    here = Path(__file__).resolve().parent
    cpp = here / "csrc" / "nvfp_emulation.cpp"
    if not cpp.is_file():
        raise FileNotFoundError(f"NVFP emulation source not found: {cpp}")

    kw = {
        "name": "nvfp_emul_jit",
        "sources": [str(cpp)],
        "verbose": bool(int(os.environ.get("NVFP_EMUL_BUILD_VERBOSE", "0"))),
    }
    if JIT_EXTRA_CFLAGS:
        kw["extra_cflags"] = list(JIT_EXTRA_CFLAGS)
    if JIT_EXTRA_LDFLAGS:
        kw["extra_ldflags"] = list(JIT_EXTRA_LDFLAGS)
    if JIT_EXTRA_INCLUDE_PATHS:
        kw["extra_include_paths"] = list(JIT_EXTRA_INCLUDE_PATHS)
    kw.update(_LOAD_KWARGS)

    return load(**kw)


def emulated_scaled_fp4_mm(*args, **kwargs):
    """Same signature as the pybind export in csrc/nvfp_emulation.cpp."""
    return _load_nvfp_emul().emulated_scaled_fp4_mm(*args, **kwargs)


RZ = 0
RNE = 1

__all__ = [
    "RZ",
    "RNE",
    "JIT_EXTRA_CFLAGS",
    "JIT_EXTRA_LDFLAGS",
    "JIT_EXTRA_INCLUDE_PATHS",
    "configure_jit",
    "emulated_scaled_fp4_mm",
]
