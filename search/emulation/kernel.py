"""
Deterministic NVFP4 emulation kernel API.
Drop-in style for ``ops.cutlass_scaled_fp4_mm`` (same tensor arguments + out_dtype).
"""
import torch
from typing import Optional

from .core import MMAEngine, RoundStrategy
from .config import HardwareConfig


class EmulationKernel:
    """
    NVFP4 MMA emulation with fixed W3 (stage 3 mantissa width), W4 (stage 4),
    and W (stage-3 fan-in, i.e. how many K-groups reduce together; hardware uses 4).
    """

    def __init__(
        self,
        config: Optional[HardwareConfig] = None,
        w_stage3: int = 34,
        w_stage4: int = 28,
        w_reduce: int = 4,
        stage3_rounding: RoundStrategy = RoundStrategy.RZ,
        stage4_rounding: RoundStrategy = RoundStrategy.RZ,
        m_chunk_size: int = 128,
    ):
        if config is not None:
            self.w_stage3 = config.w_stage3
            self.w_stage4 = config.w_stage4
            self.w_reduce = config.w_reduce
            self.stage3_rounding = config.stage3_rounding
            self.stage4_rounding = config.stage4_rounding
        else:
            self.w_stage3 = w_stage3
            self.w_stage4 = w_stage4
            self.w_reduce = w_reduce
            self.stage3_rounding = stage3_rounding
            self.stage4_rounding = stage4_rounding
        self.m_chunk_size = m_chunk_size

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        block_scale_a: torch.Tensor,
        block_scale_b: torch.Tensor,
        alpha: torch.Tensor,
        out_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return self.forward(a, b, block_scale_a, block_scale_b, alpha, out_dtype)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        block_scale_a: torch.Tensor,
        block_scale_b: torch.Tensor,
        alpha: torch.Tensor,
        out_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        M = a.shape[0]
        N = b.shape[0]
        K = a.shape[1] * 2
        result = MMAEngine.emulation_scaled_fp4_mm(
            a_fp4=a,
            b_fp4=b,
            scale_a=block_scale_a,
            scale_b=block_scale_b,
            alpha_tensor=alpha,
            M=M,
            N=N,
            K=K,
            W_stage3=self.w_stage3,
            W_stage4=self.w_stage4,
            w_reduce=self.w_reduce,
            stage3_rounding=self.stage3_rounding,
            stage4_rounding=self.stage4_rounding,
            m_chunk_size=self.m_chunk_size,
            group_size=16,
        )
        return result.to(out_dtype)

    @classmethod
    def for_rtx_5090(cls) -> "EmulationKernel":
        return cls(
            w_stage3=34,
            w_stage4=28,
            w_reduce=4,
            stage3_rounding=RoundStrategy.RZ,
            stage4_rounding=RoundStrategy.RZ,
        )

    @classmethod
    def from_config(cls, config: HardwareConfig) -> "EmulationKernel":
        return cls(config=config)


def emulated_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
    config: Optional[HardwareConfig] = None,
) -> torch.Tensor:
    kernel = EmulationKernel.from_config(config) if config else EmulationKernel.for_rtx_5090()
    return kernel(a, b, block_scale_a, block_scale_b, alpha, out_dtype)


emulated_scaled_fp4_mm = emulated_fp4_mm
