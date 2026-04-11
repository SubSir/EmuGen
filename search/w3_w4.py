import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search.emulation import (
    SearchNVFPEmulationState,
    emulate_nvfp_scaled_fp4_mm,
    nvfp_unpack_fp4_to_fp16,
    nvfp_swizzled_block_scale_to_linear,
)

import torch
from gemm_compare.backends import build_gemm_compare_fns

parser = argparse.ArgumentParser(description="Search w (backend-agnostic quant + real GEMM).")
parser.add_argument(
    "--backend",
    default="nvfp",
    choices=("mxfp", "nvfp"),
    help="mxfp: flashinfer mm_fp4; nvfp: Cutlass scaled fp4",
)
args = parser.parse_args()

GROUP_SIZE = 32 if args.backend == "mxfp" else 16
ab_dtype = torch.float16
A = torch.zeros(256, 256, dtype=ab_dtype, device="cuda")
B = torch.zeros(256, 256, dtype=ab_dtype, device="cuda")

quant_fn, real_fn, emul_fn, meta = build_gemm_compare_fns(args.backend, out_dtype=ab_dtype)

W_CANDIDATES = range(20, 50)
M_VALUES = range(0, 3699)


def set_group(X: torch.Tensor, i: int, j: float, group_size: int):
  for k in range(group_size):
    X[0, i * group_size + k] = j


def emu_matches_real(real_out: torch.Tensor, emu_out: torch.Tensor) -> bool:
  return torch.allclose(real_out, emu_out, rtol=0.0, atol=0.0)


survivors = list(W_CANDIDATES)
for M in M_VALUES:
  M = float(M)
  set_group(A, 0, M, GROUP_SIZE)
  set_group(B, 0, M, GROUP_SIZE)

  A[0, GROUP_SIZE+1] = 1.0
  B[0, GROUP_SIZE+1] = 1.0

  set_group(A, 4, -M, GROUP_SIZE)
  set_group(B, 4, M, GROUP_SIZE)

  state = quant_fn(A, B)
  real_out = real_fn(state)
  print(real_out[0, 0].item())
  if real_out[0, 0].item() ==  0.0:
    break

  next_survivors = []
  for w3 in survivors:
    state.w_stage3 = w3
    out = emulate_nvfp_scaled_fp4_mm(
      state,
      unpack_fp4=nvfp_unpack_fp4_to_fp16,
      linearize_block_scales=nvfp_swizzled_block_scale_to_linear,
    )
    if emu_matches_real(real_out, out):
      next_survivors.append(w3)
  survivors = next_survivors
  print(f"M={M} survivors={survivors}")
  if not survivors:
    break
