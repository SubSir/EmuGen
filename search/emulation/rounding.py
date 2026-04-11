"""
Rounding strategy implementations for NVFP4 emulation.
Fixed: Rounding is applied in the final cast to FP32 step.
"""
from enum import Enum
from typing import Callable, Dict
import torch


class RoundStrategy(Enum):
    """Supported rounding strategies."""
    RZ = "round_toward_zero"           # 已实现
    RNE = "round_to_nearest_even"      # 已实现 (用于最终 cast)
    RU = "round_up"                    # 留空
    RD = "round_down"                  # 留空
    RNA = "round_to_nearest_away"      # 留空


# Matches ``nvfp_cpp_emul.RZ`` / ``RNE`` (pybind passes these ints).
_NVFP_CPP_RZ = 0
_NVFP_CPP_RNE = 1


def coerce_round_strategy(rounding: RoundStrategy | int) -> RoundStrategy:
    """Accept :class:`RoundStrategy` or C++ emulator codes (0=RZ, 1=RNE)."""
    if isinstance(rounding, RoundStrategy):
        return rounding
    if rounding == _NVFP_CPP_RZ:
        return RoundStrategy.RZ
    if rounding == _NVFP_CPP_RNE:
        return RoundStrategy.RNE
    raise TypeError(
        f"Rounding must be RoundStrategy or nvfp_cpp_emul code (RZ=0, RNE=1), got {rounding!r}"
    )


class RoundingRegistry:
    """Registry for rounding strategy implementations."""
    _strategies: Dict[RoundStrategy, Callable] = {}
    
    @classmethod
    def register(cls, strategy: RoundStrategy):
        """Decorator to register a rounding implementation."""
        def decorator(func):
            cls._strategies[strategy] = func
            return func
        return decorator
    
    @classmethod
    def get(cls, strategy: RoundStrategy) -> Callable:
        """Get the implementation for a rounding strategy."""
        if strategy not in cls._strategies:
            raise NotImplementedError(
                f"Rounding strategy {strategy} is not yet implemented. "
                f"Available: {list(cls._strategies.keys())}"
            )
        return cls._strategies[strategy]
    
    @classmethod
    def is_implemented(cls, strategy: RoundStrategy) -> bool:
        """Check if a strategy is implemented."""
        return strategy in cls._strategies


# =============================================================================
# Implemented Rounding Strategies
# These are used in the final FP32 cast step
# =============================================================================

@RoundingRegistry.register(RoundStrategy.RZ)
def round_toward_zero(value_f64, target_exp=None, W=None):
    """
    Round Toward Zero (RZ) - Truncate extra precision.
    This function signature matches the registry expectation,
    but the actual implementation is in HardwareCore.to_float32_with_rounding
    """
    # This is a placeholder - actual logic is in HardwareCore
    return torch.trunc(value_f64)


@RoundingRegistry.register(RoundStrategy.RNE)
def round_to_nearest_even(value_f64, target_exp=None, W=None):
    """
    Round to Nearest Even (RNE).
    This function signature matches the registry expectation,
    but the actual implementation is in HardwareCore.to_float32_with_rounding
    """
    # This is a placeholder - actual logic is in HardwareCore
    # PyTorch's default float64->float32 cast uses RNE
    return value_f64.to(torch.float32).to(torch.float64)
