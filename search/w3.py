import argparse
import random
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "mxfp_cpp_emul"))

from gemm_compare.data import DataGenerator
from search.emulation import emulate_nvfp_scaled_fp4_mm, nvfp_unpack_fp4_to_fp16, nvfp_swizzled_block_scale_to_linear

import torch
from gemm_compare.backends import build_gemm_compare_fns
from gemm_compare.backends.mxfp import mxfp_state_as_nvfp_emulation
from mxfp import mxfp_swizzled_scale_to_linear_fp32, unpack_mxfp4_to_fp16

parser = argparse.ArgumentParser(description="Search w (backend-agnostic quant + real GEMM).")
parser.add_argument(
    "--backend",
    default="mxfp",
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

SHAPE = (1024, 128)
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


def _sentinel_mask(t: torch.Tensor) -> torch.Tensor:
  """NaN, ±Inf, and exact 0 (including -0) are treated as one equivalence class vs finite non-zeros."""
  return torch.isnan(t) | torch.isinf(t) | (t == 0)


def emu_matches_real(real_out: torch.Tensor, emu_out: torch.Tensor) -> bool:
  if real_out.shape != emu_out.shape:
    return False
  r, e = real_out, emu_out
  sr, se = _sentinel_mask(r), _sentinel_mask(e)
  # Match if both are sentinel-like (0 / nan / inf in any combination), or both finite non-zero and exactly equal.
  ok = (sr & se) | (~sr & ~se & torch.eq(r, e))
  return bool(torch.all(ok).item())


def _flat_index_to_coords(flat_i: int, shape: tuple[int, ...]) -> tuple[int, ...]:
  coords: list[int] = []
  n = flat_i
  for s in reversed(shape):
    coords.append(n % s)
    n //= s
  return tuple(reversed(coords))


def print_w3_eliminated(w3: int, real_out: torch.Tensor, emu_out: torch.Tensor) -> None:
  """When a candidate fails ``emu_matches_real``, print why / max |err| on finite non-sentinel pairs."""
  if real_out.shape != emu_out.shape:
    print(f"  eliminate w3={w3}: shape mismatch real={tuple(real_out.shape)} emu={tuple(emu_out.shape)}")
    return

  r = real_out.detach()
  e = emu_out.detach()
  sr, se = _sentinel_mask(r), _sentinel_mask(e)
  ok = (sr & se) | (~sr & ~se & torch.eq(r, e))
  if bool(torch.all(ok).item()):
    return

  xor_class = sr ^ se
  if xor_class.any():
    nz = xor_class.nonzero()
    idx = tuple(int(x) for x in nz[0].tolist())
    print(
      f"  eliminate w3={w3}: sentinel vs finite mismatch | index={idx} | "
      f"real={r[idx].item()!r} emu={e[idx].item()!r} "
      f"(sentinel real={bool(sr[idx].item())} emu={bool(se[idx].item())})"
    )

  strict_finite = ~sr & ~se
  neq = strict_finite & ~torch.eq(r, e)
  if neq.any():
    d = (r.float() - e.float()).abs()
    d = torch.where(neq, d, torch.zeros_like(d))
    flat = d.reshape(-1)
    flat_i = int(flat.argmax().item())
    max_err = float(flat[flat_i].item())
    idx = _flat_index_to_coords(flat_i, tuple(r.shape))
    rv = r[idx].item()
    ev = e[idx].item()
    print(
      f"  eliminate w3={w3}: max_abs_err={max_err:.6g} | index={idx} | real={rv!r} | emu={ev!r}"
    )

W_CANDIDATES = range(20, 50)
survivors = list(W_CANDIDATES)

while True:
  da, db = random.choice(DISTRIBUTIONS), random.choice(DISTRIBUTIONS)
  A = DataGenerator.get_random_tensor(SHAPE, da, device=DEVICE, dtype=ab_dtype)
  B = DataGenerator.get_random_tensor(SHAPE, db, device=DEVICE, dtype=ab_dtype)

  A[:, 64:] = 0
  B[:, 64:] = 0

  state = quant_fn(A, B)
  real_out = real_fn(state)
  next_survivors = []
  for w3 in survivors:
    state.w_stage3 = w3
    if args.backend == "nvfp":
      out = emulate_nvfp_scaled_fp4_mm(
        state,
        unpack_fp4=nvfp_unpack_fp4_to_fp16,
        linearize_block_scales=nvfp_swizzled_block_scale_to_linear,
      )
    else:
      out = emulate_nvfp_scaled_fp4_mm(
        mxfp_state_as_nvfp_emulation(state),
        unpack_fp4=unpack_mxfp4_to_fp16,
        linearize_block_scales=mxfp_swizzled_scale_to_linear_fp32,
      )
    if emu_matches_real(real_out, out):
      next_survivors.append(w3)
    else:
      print_w3_eliminated(w3, real_out, out)
  survivors = next_survivors
  print(f"survivors={survivors} | A={da} B={db}")
  if len(survivors) <= 1:
    break
