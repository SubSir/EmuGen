"""Generic real vs emulation GEMM comparison loop."""
from __future__ import annotations

import random
import sys
import time
import traceback
from typing import Any, Callable, Type

import torch

from .data import DataGenerator

QuantFn = Callable[[torch.Tensor, torch.Tensor], Any]
GemmFn = Callable[[Any], torch.Tensor]


def compare_tensors(
    real_output: torch.Tensor,
    modeling_res: torch.Tensor,
    *,
    sample_count: int = 5,
) -> dict[str, Any]:
    """Element-wise comparison (exact match + NaN agreement), same rules as legacy test.py."""
    diff = (real_output.float() - modeling_res.float()).abs()
    matches = real_output == modeling_res
    diff[matches] = 0.0
    both_nan = real_output.isnan() & modeling_res.isnan()
    diff[both_nan] = 0.0
    diff[diff.isnan()] = float("inf")

    max_diff = diff.max().item()
    status = "SUCCESS" if max_diff == 0 else "MISMATCH"
    mismatch_details: list[dict[str, Any]] = []

    if status == "MISMATCH":
        mismatch_indices = torch.nonzero(diff != 0, as_tuple=False)
        for idx in mismatch_indices[:sample_count]:
            r, c = idx[0].item(), idx[1].item()
            mismatch_details.append(
                {
                    "idx": (r, c),
                    "real": real_output[r, c].item(),
                    "model": modeling_res[r, c].item(),
                    "diff": diff[r, c].item(),
                }
            )

    return {
        "status": status,
        "max_diff": max_diff,
        "mismatch_details": mismatch_details,
    }


def run_test_case(
    iter_idx: int,
    M: int,
    N: int,
    K: int,
    dist_a: str,
    dist_b: str,
    quant_fn: QuantFn,
    real_fn: GemmFn,
    emul_fn: GemmFn,
    *,
    sample_count: int = 5,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    data_gen: Type[DataGenerator] = DataGenerator,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    a = data_gen.get_random_tensor((M, K), dist_a, device=device, dtype=dtype)
    b = data_gen.get_random_tensor((N, K), dist_b, device=device, dtype=dtype)

    state = quant_fn(a, b)
    real_output = real_fn(state)
    modeling_res = emul_fn(state)

    cmp = compare_tensors(real_output, modeling_res, sample_count=sample_count)
    out: dict[str, Any] = {
        "iter": iter_idx,
        "M": M,
        "N": N,
        "K": K,
        "dist_a": dist_a,
        "dist_b": dist_b,
        **cmp,
    }
    if extra_meta:
        out["extra"] = extra_meta
    return out


def run_suite(
    quant_fn: QuantFn,
    real_fn: GemmFn,
    emul_fn: GemmFn,
    *,
    name: str = "GEMM compare",
    num_iterations: int = 100,
    distributions: list[str] | None = None,
    dims_m: list[int] | None = None,
    dims_n: list[int] | None = None,
    dims_k: list[int] | None = None,
    sample_count: int = 5,
    max_mismatch_print: int = 20,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    data_gen: Type[DataGenerator] = DataGenerator,
    seed: int | None = None,
) -> int:
    """
    Randomized suite like legacy test.py main(). Returns 0 if all exact matches, else 1.
    """
    if seed is None:
        seed = int(time.time() * 1e6) ^ int.from_bytes(__import__("os").urandom(8), "little")
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print("seed =", seed)

    distributions = distributions or [
        "normal",
        "uniform",
        "large",
        "outliers",
        "mixed_rows",
        "abs_large",
    ]
    dims_m = dims_m or [128, 256, 1024, 2048]
    dims_n = dims_n or [128, 256, 1024, 2048, 4096]
    dims_k = dims_k or [128, 256, 512, 1024]

    print(f"\n{'=' * 70}")
    print(name)
    print(f"{'=' * 70}")
    print(f"Iterations: {num_iterations}")
    print(f"Distributions: {distributions}")
    print(f"Dims M: {dims_m} | N: {dims_n} | K: {dims_k}")
    print("=" * 70)

    results: list[dict[str, Any]] = []
    mismatch_print_count = 0

    for i in range(num_iterations):
        M = random.choice(dims_m)
        N = random.choice(dims_n)
        K = random.choice(dims_k)
        da, db = random.choice(distributions), random.choice(distributions)

        print(f"\rTest {i + 1}/{num_iterations}: K={K:4}, A={da:10}, B={db:10} ... ", end="")
        sys.stdout.flush()

        try:
            res = run_test_case(
                i,
                M,
                N,
                K,
                da,
                db,
                quant_fn,
                real_fn,
                emul_fn,
                sample_count=sample_count,
                device=device,
                dtype=dtype,
                data_gen=data_gen,
            )
            results.append(res)
            print(res["status"])

            if res["status"] == "MISMATCH" and mismatch_print_count < max_mismatch_print:
                mismatch_print_count += 1
                print("    >>> Mismatch Details:")
                for d in res["mismatch_details"]:
                    print(
                        f"      Pos {d['idx']}: Real={d['real']:.6f} | "
                        f"Model={d['model']:.6f} | Diff={d['diff']:.6f}"
                    )
        except Exception as e:
            print(f"\nFATAL ERROR on Test {i + 1}: {e}")
            traceback.print_exc()

    mismatches = [r for r in results if r["status"] == "MISMATCH"]
    print("\n" + "=" * 70)
    print(f"SUMMARY | {name}")
    print(f"Total: {num_iterations}, Matches: {len(results) - len(mismatches)}, Mismatches: {len(mismatches)}")
    print("=" * 70)

    if len(mismatches) == 0:
        print("*** All tests: exact bitwise match ***")
        return 0
    print(f"Exact-match rate: {(len(results) - len(mismatches)) / max(len(results), 1) * 100:.2f}%")
    return 1
