#!/usr/bin/env python3
"""CLI: run randomized real vs emul suite for NVFP or MXFP."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on sys.path for `python gemm_compare/cli.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description="GEMM real vs emulation compare")
    parser.add_argument(
        "backend",
        choices=("nvfp", "mxfp"),
        help="nvfp: cutlass vs C++ emul | mxfp: flashinfer mm_fp4 vs dequant@matmul",
    )
    parser.add_argument("-n", "--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--w-stage3", type=int, default=25, help="NVFP emul only")
    parser.add_argument("--w-stage4", type=int, default=25, help="NVFP emul only")
    parser.add_argument("--nvfp-out", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--mxfp-backend", default="cudnn", help="flashinfer mm_fp4 backend")
    parser.add_argument("--group-size", type=int, default=32, help="MXFP block size")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA required for these backends.", file=sys.stderr)
        return 2

    from gemm_compare.runner import run_suite

    if args.backend == "nvfp":
        from gemm_compare.backends.nvfp import build_nvfp_fns

        out_dtype = torch.float16 if args.nvfp_out == "float16" else torch.bfloat16
        quant_fn, real_fn, emul_fn, meta = build_nvfp_fns(
            out_dtype=out_dtype,
            w_stage3=args.w_stage3,
            w_stage4=args.w_stage4,
        )
        name = f"NVFP | {meta}"
        dtype = torch.float16
    else:
        from gemm_compare.backends.mxfp import build_mxfp_fns

        quant_fn, real_fn, emul_fn, meta = build_mxfp_fns(
            group_size=args.group_size,
            out_dtype=torch.bfloat16,
            mm_backend=args.mxfp_backend,
        )
        name = f"MXFP | {meta}"
        dtype = torch.bfloat16

    return run_suite(
        quant_fn,
        real_fn,
        emul_fn,
        name=name,
        num_iterations=args.iterations,
        device=args.device,
        dtype=dtype,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
