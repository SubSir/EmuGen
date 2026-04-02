#!/usr/bin/env python3
"""
Benchmark ``nvfp.ops.cutlass_scaled_fp4_mm`` (hw) vs ``nvfp_cpp_emul.emulated_scaled_fp4_mm`` (emul).

Same tensor setup as ``example.py`` (pytorch_nvfp4_quantize + uint8 views).

Run (with your env)::

    conda activate llm_inference
    python nvfp_cpp_emul/speed_bench_hw_vs_emul.py
    python nvfp_cpp_emul/speed_bench_hw_vs_emul.py -M 512 -N 512 -K 256 --repeats 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXAMPLE_ROOT = Path(__file__).resolve().parent
if str(_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_ROOT))


def main() -> None:
    import torch
    import torch.utils.benchmark as benchmark

    parser = argparse.ArgumentParser(description="HW vs C++ emulation FP4 scaled GEMM timing")
    parser.add_argument("-M", type=int, default=128, help="A rows")
    parser.add_argument("-N", type=int, default=256, help="B rows (output cols)")
    parser.add_argument("-K", type=int, default=64, help="inner dim (multiple of 16; G=K/16 multiple of 4)")
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations per path")
    parser.add_argument("--repeats", type=int, default=100, help="timer repeats (torch.utils.benchmark)")
    parser.add_argument(
        "--out-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="cutlass_scaled_fp4_mm output dtype",
    )
    parser.add_argument(
        "--m-chunk-size",
        type=int,
        default=128,
        help="emulated_scaled_fp4_mm m_chunk_size (must match example if comparing accuracy)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required.", file=sys.stderr)
        sys.exit(1)

    M, N, K = args.M, args.N, args.K
    if K % 16 != 0 or (K // 16) % 4 != 0:
        print("K must be divisible by 16 and K/16 must be divisible by 4.", file=sys.stderr)
        sys.exit(1)

    out_dtype = torch.float16 if args.out_dtype == "float16" else torch.bfloat16
    device = "cuda"

    import nvfp.ops as ops
    import nvfp.pseudo_quant as pseudo_quant
    from nvfp_cpp_emul import RZ, emulated_scaled_fp4_mm

    FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX = 6.0, 448.0
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(N, K, device=device, dtype=torch.float16)

    def global_scale(t: torch.Tensor) -> torch.Tensor:
        amax = torch.abs(t).max().to(torch.float32).item()
        g = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (amax if amax > 0 else 1.0)
        return torch.tensor([g], device=device, dtype=torch.float32)

    a_gs, b_gs = global_scale(a), global_scale(b)
    alpha = torch.tensor(
        [1.0 / (a_gs.item() * b_gs.item())], device=device, dtype=torch.float32
    )

    a_fp4, scale_a = pseudo_quant.pytorch_nvfp4_quantize(a, a_gs)
    b_fp4, scale_b = pseudo_quant.pytorch_nvfp4_quantize(b, b_gs)
    a_fp4 = a_fp4.contiguous().view(torch.uint8)
    b_fp4 = b_fp4.contiguous().view(torch.uint8)

    def run_hw():
        return ops.cutlass_scaled_fp4_mm(
            a_fp4, b_fp4, scale_a, scale_b, alpha, out_dtype
        )

    def run_emul():
        return emulated_scaled_fp4_mm(
            a_fp4,
            b_fp4,
            scale_a,
            scale_b,
            alpha,
            w_stage3=25,
            w_stage4=25,
            m_chunk_size=args.m_chunk_size,
            stage3_rounding=RZ,
            stage4_rounding=RZ,
        )

    # Warmup (includes JIT compile on first emul call)
    for _ in range(args.warmup):
        _ = run_hw()
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        _ = run_emul()
    torch.cuda.synchronize()

    t_hw = benchmark.Timer(
        stmt="fn()",
        globals={"fn": run_hw},
        label="cutlass_scaled_fp4_mm (hw)",
        sub_label=f"M={M} N={N} K={K}",
        num_threads=torch.get_num_threads(),
    ).timeit(args.repeats)

    t_emul = benchmark.Timer(
        stmt="fn()",
        globals={"fn": run_emul},
        label="emulated_scaled_fp4_mm (C++ JIT)",
        sub_label=f"M={M} N={N} K={K}",
        num_threads=torch.get_num_threads(),
    ).timeit(args.repeats)

    ms_hw = t_hw.mean * 1e3
    ms_emul = t_emul.mean * 1e3
    flops = 2.0 * M * N * K

    def _line(tag: str, m) -> None:
        print(
            f"{tag}: mean={m.mean * 1e3:.4f} ms  median={m.median * 1e3:.4f} ms  "
            f"IQR={m.iqr * 1e3:.4f} ms"
        )

    _line("hw (cutlass_scaled_fp4_mm)", t_hw)
    _line("emul (emulated_scaled_fp4_mm)", t_emul)
    print()
    print(f"mean latency: hw {ms_hw:.4f} ms  |  emul {ms_emul:.4f} ms")
    if ms_hw > 0 and ms_emul > 0:
        if ms_emul >= ms_hw:
            print(f"hw is {ms_emul / ms_hw:.2f}x faster than emul (mean time)")
        else:
            print(f"emul is {ms_hw / ms_emul:.2f}x faster than hw (mean time)")
    print(f"throughput (hw):   {flops / t_hw.mean / 1e12:.3f} TFLOP/s")
    print(f"throughput (emul): {flops / t_emul.mean / 1e12:.3f} TFLOP/s")


if __name__ == "__main__":
    main()
