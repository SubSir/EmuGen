import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

M = 28672.0
GROUP_SIZE = 32 if args.backend == "mxfp" else 16
ab_dtype = torch.float16
A = torch.zeros(256, 256, dtype=ab_dtype, device="cuda")
B = torch.zeros(256, 256, dtype=ab_dtype, device="cuda")

quant_fn, real_fn, emul_fn, meta = build_gemm_compare_fns(args.backend)


def set_group(X: torch.Tensor, i: int, j: float, group_size: int):
  for k in range(group_size):
    X[0, i * group_size + k] = j

set_group(A, 0, M, GROUP_SIZE)
set_group(B, 0, M, GROUP_SIZE)

set_group(A, 1, 1.0, GROUP_SIZE)
set_group(B, 1, 1.0, GROUP_SIZE)

set_group(A, 2, M, GROUP_SIZE)
set_group(B, 2, -M, GROUP_SIZE)

state = quant_fn(A, B)

real_out = real_fn(state)

if real_out[0, 0].item() == GROUP_SIZE * 1.0: # w >=3
  A[:, :] = 0
  B[:, :] = 0

  A[0, :] = 1
  B[0, :] = 1

  set_group(A, 0, M, GROUP_SIZE)
  set_group(B, 0, M, GROUP_SIZE)

  set_group(A, 1, -M, GROUP_SIZE)
  set_group(B, 1, M, GROUP_SIZE)

  state = quant_fn(A, B)
  base_real = real_fn(state)

  for i in range(2, 256 // GROUP_SIZE):
    A[:, :] = 0
    B[:, :] = 0

    A[0, :] = 1
    B[0, :] = 1

    set_group(A, 0, M, GROUP_SIZE)
    set_group(B, 0, M, GROUP_SIZE)

    set_group(A, i, -M, GROUP_SIZE)
    set_group(B, i, M, GROUP_SIZE)

    state = quant_fn(A, B)
    real_out = real_fn(state)
    if real_out[0, 0].item() != base_real[0, 0].item():
      print("w == {}".format(i)) # note: if w == 2 (mxfp), it will also print 2, you can have try it yourself
      break

else: # w <= 2

  A[:, :] = 0
  B[:, :] = 0

  set_group(A, 0, 1, GROUP_SIZE)
  set_group(B, 0, 1, GROUP_SIZE)

  set_group(A, 2, M, GROUP_SIZE)
  set_group(A, 3, M, GROUP_SIZE)
  set_group(B, 2, M, GROUP_SIZE)
  set_group(B, 3, -M, GROUP_SIZE)

  state = quant_fn(A, B)
  real_out = real_fn(state)

  if real_out[0, 0].item() == GROUP_SIZE * 1.0:
    print("w == 2")
  else:
    print("w == 1")
