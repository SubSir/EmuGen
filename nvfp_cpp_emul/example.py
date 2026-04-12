#!/usr/bin/env python3
"""
Example: NVFP4 scaled GEMM emulation via JIT-compiled libtorch extension (no pip install).

Run as ``python nvfp_cpp_emul/example.py`` from the repo root (the script adds its
parent directory to ``sys.path``). First call JIT-compiles ``csrc/nvfp_emulation.cpp``.

Optional compile flags before first use::

  import nvfp_cpp_emul
  nvfp_cpp_emul.JIT_EXTRA_CFLAGS.append("-O3")

Same logical pipeline as nvfp_kernel/verify_acc_modeling.py:
  unpack FP4 -> FP16 group dots -> apply block scales -> fp64 sum per K64 groups
  -> optional RZ/RNE to fp32 -> multiply alpha -> FP16.

If ``nvfp.ops`` is importable (your CUDA nvfp stack), this script also compares against
``cutlass_scaled_fp4_mm`` like the original verifier.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Parent of the inner package (directory that contains ``nvfp_cpp_emul/``).
_EXAMPLE_ROOT = Path(__file__).resolve().parent
if str(_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_ROOT))


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        print("CUDA is required (tensors must match GPU quant layout).", file=sys.stderr)
        sys.exit(1)

    from nvfp_cpp_emul import RZ, emulated_scaled_fp4_mm

    device = "cuda"
    M, N, K = 128, 256, 64
    torch.manual_seed(0)

    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(N, K, device=device, dtype=torch.float16)

    FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX = 6.0, 448.0

    def global_scale(t: torch.Tensor) -> torch.Tensor:
        amax = torch.abs(t).max().to(torch.float32).item()
        g = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (amax if amax > 0 else 1.0)
        return torch.tensor([g], device=device, dtype=torch.float32)

    a_gs, b_gs = global_scale(a), global_scale(b)
    alpha = torch.tensor([1.0 / (a_gs.item() * b_gs.item())], device=device, dtype=torch.float32)

    import nvfp.ops as ops
    import nvfp.pseudo_quant as pseudo_quant

    a_fp4, scale_a = pseudo_quant.pytorch_nvfp4_quantize(a, a_gs)
    b_fp4, scale_b = pseudo_quant.pytorch_nvfp4_quantize(b, b_gs)

    a_fp4 = a_fp4.contiguous().view(torch.uint8)
    b_fp4 = b_fp4.contiguous().view(torch.uint8)
    hw = ops.cutlass_scaled_fp4_mm(
        a_fp4, b_fp4, scale_a, scale_b, alpha, torch.float16
    )

    # Match verify_acc_modeling.py simple model: W=25, pure RZ, sum across K64 blocks.
    emul = emulated_scaled_fp4_mm(
        a_fp4,
        b_fp4,
        scale_a,
        scale_b,
        alpha,
        m_chunk_size=128,
        stage3_rounding=RZ,
        stage4_rounding=RZ,
    )

    print("emulation output:", emul.shape, emul.dtype, emul.device)
    if ops is not None:
        diff = (hw.float() - emul.float()).abs()
        match = hw == emul
        diff = torch.where(match, torch.zeros_like(diff), diff)
        print("max |hw - emul|:", diff.max().item())


if __name__ == "__main__":
    main()
