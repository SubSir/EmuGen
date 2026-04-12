"""
Core emulation classes: HardwareCore and MMAEngine.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from .rounding import RoundStrategy, coerce_round_strategy
from .utils import nvfp_swizzled_block_scale_to_linear, nvfp_unpack_fp4_to_fp16


class HardwareCore:
    """Hardware-style reduction and accumulation with configurable rounding."""

    @staticmethod
    def to_float32_with_rounding(tensor_f64, rounding=RoundStrategy.RZ):
        if rounding == RoundStrategy.RZ:
            f32 = tensor_f64.to(torch.float32)
            mask = f32.abs() > tensor_f64.abs()
            res = f32.clone()
            if mask.any():
                res[mask] = torch.nextafter(f32[mask], torch.zeros_like(f32[mask]))
            return res
        if rounding == RoundStrategy.RNE:
            return tensor_f64.to(torch.float32)
        raise NotImplementedError(f"Rounding {rounding} not implemented in to_float32_with_rounding")

    @staticmethod
    def hardware_add_wbits(acc_fp32, new_val_wbits, W=25, rounding=RoundStrategy.RZ):
        """Stage 4: FP32 accumulator + lane value -> FP32 (``W`` retained for API compatibility; unused)."""
        sum_f64 = acc_fp32.double() + new_val_wbits.double()
        return HardwareCore.to_float32_with_rounding(sum_f64, rounding)

    @staticmethod
    def hardware_reduction_nto1(v_list, W=25, output_fp32=True, rounding=RoundStrategy.RZ):
        """Stage 3: n-to-1 reduction — fp64 sum per lane (``W`` retained for API compatibility; unused)."""
        v_aligned_sum = torch.zeros_like(v_list[0], dtype=torch.float64)
        for v in v_list:
            v_aligned_sum += v.double()
        summed_f64 = v_aligned_sum
        if output_fp32:
            return HardwareCore.to_float32_with_rounding(summed_f64, rounding)
        return summed_f64

    @staticmethod
    def hardware_reduction_4to1(v_list, W=25, output_fp32=True, rounding=RoundStrategy.RZ):
        """Backward-compatible name for 4-wide reduction (same as hardware_reduction_nto1)."""
        return HardwareCore.hardware_reduction_nto1(v_list, W, output_fp32, rounding)


class MMAEngine:
    """NVFP4 scaled GEMM emulation (Python reference)."""

    @staticmethod
    def stage1_inner_mma_fp16(val_a, val_b, group_size: int = 16):
        M, K = val_a.shape
        N, _ = val_b.shape
        if K % group_size != 0:
            raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")
        K_groups = K // group_size
        a_grouped = val_a.view(M, K_groups, group_size).to(torch.float16)
        b_grouped = val_b.view(N, K_groups, group_size).to(torch.float16)
        partial_sum1 = torch.einsum("mgk,ngk->mgn", a_grouped, b_grouped).to(torch.float16)
        return partial_sum1.permute(0, 2, 1)

    @staticmethod
    def emulation_scaled_fp4_mm(
        a_fp4,
        b_fp4,
        scale_a,
        scale_b,
        alpha_tensor,
        M,
        N,
        K,
        W_stage3=25,
        W_stage4=25,
        w_reduce=4,
        stage3_rounding: RoundStrategy | int = RoundStrategy.RZ,
        stage4_rounding: RoundStrategy | int = RoundStrategy.RZ,
        m_chunk_size=128,
        group_size: int = 16,
        unpack_fp4: Optional[Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]] = None,
        linearize_block_scales: Optional[Callable[[torch.Tensor, int, int], torch.Tensor]] = None,
    ):
        """Scaled FP4 MMA emulation with M-chunking. ``group_size`` is the K tile for stage-1 dots (NVFP 16, MXFP 32)."""
        stage3_rounding = coerce_round_strategy(stage3_rounding)
        stage4_rounding = coerce_round_strategy(stage4_rounding)
        unpack = unpack_fp4 or nvfp_unpack_fp4_to_fp16
        linearize = linearize_block_scales or nvfp_swizzled_block_scale_to_linear

        assert K % group_size == 0, f"K must be multiple of group_size ({group_size})"
        G = K // group_size
        assert G % w_reduce == 0, f"G=K/group_size ({G}) must be divisible by w_reduce ({w_reduce})"
        num_blocks = G // w_reduce

        val_b_fp16 = unpack(b_fp4, (N, K))
        s_b = linearize(scale_b, N, G)
        s_a_all = linearize(scale_a, M, G)
        val_a_fp16_all = unpack(a_fp4, (M, K))

        summed_results_list = []
        for m_start in range(0, M, m_chunk_size):
            m_end = min(m_start + m_chunk_size, M)
            curr_chunk_size = m_end - m_start
            val_a_chunk = val_a_fp16_all[m_start:m_end, :]
            s_a_chunk = s_a_all[m_start:m_end, :]

            ps1_chunk = MMAEngine.stage1_inner_mma_fp16(
                val_a_chunk, val_b_fp16, group_size=group_size
            )
            if group_size == 32:
                s_a_chunk = s_a_chunk.double() # MXFP needs double precision for scale alignment
            combined_scales_chunk = s_a_chunk.unsqueeze(1) * s_b.unsqueeze(0)
            scaled_partials_chunk = ps1_chunk.float() * combined_scales_chunk

            scaled_grouped = scaled_partials_chunk.view(curr_chunk_size, N, num_blocks, w_reduce)
            v_list = [scaled_grouped[..., i] for i in range(w_reduce)]

            if num_blocks == 1:
                summed_groups = HardwareCore.hardware_reduction_nto1(
                    v_list, W=W_stage3, output_fp32=True, rounding=stage3_rounding
                )
            else:
                summed_groups = HardwareCore.hardware_reduction_nto1(
                    v_list, W=W_stage3, output_fp32=False, rounding=stage3_rounding
                )

            if num_blocks == 1:
                summed_chunk = summed_groups.squeeze(-1)
            else:
                acc = HardwareCore.to_float32_with_rounding(summed_groups[..., 0], stage4_rounding)
                for i in range(1, num_blocks):
                    acc = HardwareCore.hardware_add_wbits(
                        acc,
                        summed_groups[..., i],
                        W=W_stage4,
                        rounding=stage4_rounding,
                    )
                summed_chunk = acc

            summed_results_list.append(summed_chunk)

        summed_result = torch.cat(summed_results_list, dim=0)
        alpha_val = alpha_tensor.item()
        return (summed_result * alpha_val).to(torch.float16)

