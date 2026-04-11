"""
Python NVFP4 MMA emulation (accuracy model) and helpers.

Quick start:
    >>> from search.emulation import emulated_fp4_mm, EmulationKernel
    >>> out = kernel(a_fp4, b_fp4, scale_a, scale_b, alpha, torch.float16)

Pre-quantized state + emulation only:
    >>> from search.emulation import SearchNVFPEmulationState, emulate_nvfp_scaled_fp4_mm
    >>> out = emulate_nvfp_scaled_fp4_mm(state, unpack_fp4=my_unpack, linearize_block_scales=my_linearize)
"""
from .core import HardwareCore, MMAEngine
from .utils import (
    NVFP4Utils,
    nvfp_unpack_fp4_to_fp16,
    nvfp_swizzled_block_scale_to_linear,
)
from .rounding import RoundStrategy, RoundingRegistry, coerce_round_strategy
from .config import HardwareConfig, CONFIG_RTX_5090
from .kernel import EmulationKernel, emulated_fp4_mm, emulated_scaled_fp4_mm
from .gemm import SearchNVFPEmulationState, emulate_nvfp_scaled_fp4_mm

__all__ = [
    "HardwareCore",
    "MMAEngine",
    "NVFP4Utils",
    "nvfp_unpack_fp4_to_fp16",
    "nvfp_swizzled_block_scale_to_linear",
    "RoundStrategy",
    "coerce_round_strategy",
    "RoundingRegistry",
    "HardwareConfig",
    "CONFIG_RTX_5090",
    "EmulationKernel",
    "emulated_fp4_mm",
    "emulated_scaled_fp4_mm",
    "SearchNVFPEmulationState",
    "emulate_nvfp_scaled_fp4_mm",
]
