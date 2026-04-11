import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemm_compare.data import DataGenerator
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
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="RNG seed (default: time/urandom, same style as gemm_compare.runner)",
)
args = parser.parse_args()

DISTRIBUTIONS = [
    "normal",
    "uniform",
    "large",
    "outliers",
    "mixed_rows",
    "abs_large",
]

SHAPE = (1024, 64)
DEVICE = "cuda"

seed = args.seed
if seed is None:
    seed = int(time.time() * 1e6) ^ int.from_bytes(__import__("os").urandom(8), "little")
random.seed(seed)
torch.manual_seed(seed)
if DEVICE == "cuda" and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
print("seed =", seed)

GROUP_SIZE = 32 if args.backend == "mxfp" else 16
ab_dtype = torch.float16

quant_fn, real_fn, emul_fn, meta = build_gemm_compare_fns(args.backend, out_dtype=ab_dtype)

def emu_matches_real(real_out: torch.Tensor, emu_out: torch.Tensor) -> bool:
  nan_real = torch.isnan(real_out)
  nan_emu = torch.isnan(emu_out)
  if not torch.equal(nan_real, nan_emu):
    return False
  real_out = real_out[~nan_real]
  emu_out = emu_out[~nan_emu]

  inf_real = torch.isinf(real_out)
  inf_emu = torch.isinf(emu_out)
  if not torch.equal(inf_real, inf_emu):
    return False
  # Same positions are inf; still require same sign (isinf mask alone is not enough).
  if inf_real.any() and not torch.equal(
    torch.sign(real_out[inf_real]), torch.sign(emu_out[inf_emu])
  ):
    return False
  real_out = real_out[~inf_real]
  emu_out = emu_out[~inf_emu]

  return torch.allclose(real_out, emu_out, rtol=0.0, atol=0.0)

W_CANDIDATES = range(33, 50)
survivors = list(W_CANDIDATES)

while True:
  da, db = random.choice(DISTRIBUTIONS), random.choice(DISTRIBUTIONS)
  A = DataGenerator.get_random_tensor(SHAPE, da, device=DEVICE, dtype=ab_dtype)
  B = DataGenerator.get_random_tensor(SHAPE, db, device=DEVICE, dtype=ab_dtype)

  state = quant_fn(A, B)
  real_out = real_fn(state)

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
  print(f"survivors={survivors} | A={da} B={db}")
  if len(survivors) <= 1:
    break
