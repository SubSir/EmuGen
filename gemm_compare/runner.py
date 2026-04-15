"""Generic real vs emulation GEMM comparison loop."""
from __future__ import annotations

import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Type

import torch

from .data import DataGenerator
from .rollout import load_rollout_artifact, quant_state_from_dict, quant_state_to_cpu_dict, save_rollout_artifact

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


def compare_mse(
    reference: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Any]:
    """Squared error stats, plus ||reference||_2 for normalization."""
    ref_f = reference.float()
    sq = (ref_f - target.float()).pow(2)
    sq = torch.nan_to_num(sq, nan=0.0, posinf=0.0, neginf=0.0)
    sq_sum = sq.sum().item()
    sq_numel = sq.numel()
    ref_sq_sum = torch.nan_to_num(ref_f.pow(2), nan=0.0, posinf=0.0, neginf=0.0).sum().item()
    per_gemm_metric = (sq_sum ** 0.5) / max(ref_sq_sum ** 0.5, 1e-12)
    return {
        "status": "OK",
        "mean_mse": sq_sum / max(sq_numel, 1),
        "max_mse": sq.max().item(),
        "mse_sum": sq_sum,
        "mse_numel": sq_numel,
        "real_sq_sum": ref_sq_sum,
        "per_gemm_rel_l2_error": per_gemm_metric,
        "mismatch_details": [],
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
    metric_mode: str = "exact",
) -> dict[str, Any]:
    a = data_gen.get_random_tensor((M, K), dist_a, device=device, dtype=dtype)
    b = data_gen.get_random_tensor((N, K), dist_b, device=device, dtype=dtype)

    state = quant_fn(a, b)
    real_output = real_fn(state)
    modeling_res = emul_fn(state)

    if metric_mode == "exact":
        cmp = compare_tensors(real_output, modeling_res, sample_count=sample_count)
    elif metric_mode == "mse":
        cmp = compare_mse(real_output, modeling_res)
    else:
        raise ValueError(f"unknown metric_mode={metric_mode!r}")
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


def run_test_case_export(
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
    backend: str,
    sample_count: int = 5,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    data_gen: Type[DataGenerator] = DataGenerator,
    verify_local: bool = False,
    include_inputs: bool = False,
) -> dict[str, Any]:
    """Quantize + real GEMM; optionally verify emulation on the export machine."""
    a = data_gen.get_random_tensor((M, K), dist_a, device=device, dtype=dtype)
    b = data_gen.get_random_tensor((N, K), dist_b, device=device, dtype=dtype)

    state = quant_fn(a, b)
    real_output = real_fn(state)
    local_check: dict[str, Any] | None = None
    if verify_local:
        modeling_res = emul_fn(state)
        local_check = compare_tensors(real_output, modeling_res, sample_count=sample_count)

    state_dict = quant_state_to_cpu_dict(state, backend, include_inputs=include_inputs)
    return {
        "iter": iter_idx,
        "M": M,
        "N": N,
        "K": K,
        "dist_a": dist_a,
        "dist_b": dist_b,
        "state": state_dict,
        "real_output": real_output.detach().cpu().contiguous(),
        "local_check": local_check,
    }


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
    metric_mode: str = "exact",
    print_iter_status: bool = True,
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
    dims_m = dims_m or [128, 256, 1024, 2048, 4096]
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
                metric_mode=metric_mode,
            )
            results.append(res)
            if print_iter_status:
                print(res["status"])
            else:
                print("OK")

            if (
                metric_mode == "exact"
                and res["status"] == "MISMATCH"
                and mismatch_print_count < max_mismatch_print
            ):
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

    print("\n" + "=" * 70)
    print(f"SUMMARY | {name}")
    print("=" * 70)
    if metric_mode == "mse":
        total_numel = sum(int(r.get("mse_numel", 0)) for r in results)
        total_mse_sum = sum(float(r.get("mse_sum", 0.0)) for r in results)
        mean_mse = total_mse_sum / max(total_numel, 1)
        per_gemm_scores = [float(r.get("per_gemm_rel_l2_error", 0.0)) for r in results]
        mean_per_gemm_score = sum(per_gemm_scores) / max(len(per_gemm_scores), 1)
        worst_iter_max = max((float(r.get("max_mse", 0.0)) for r in results), default=0.0)
        print(f"Total: {num_iterations}, Completed: {len(results)}")
        print(f"MSE mean: {mean_mse:.8e}")
        print(f"mean over GEMMs of sqrt(sum(error^2)) / sqrt(sum(real^2)): {mean_per_gemm_score:.8e}")
        print(f"MSE max (worst element): {worst_iter_max:.8e}")
        return 0

    mismatches = [r for r in results if r["status"] == "MISMATCH"]
    print(f"Total: {num_iterations}, Matches: {len(results) - len(mismatches)}, Mismatches: {len(mismatches)}")
    print("=" * 70)
    if len(mismatches) == 0:
        print("*** All tests: exact bitwise match ***")
        return 0
    print(f"Exact-match rate: {(len(results) - len(mismatches)) / max(len(results), 1) * 100:.2f}%")
    return 1


def run_suite_export(
    quant_fn: QuantFn,
    real_fn: GemmFn,
    emul_fn: GemmFn,
    *,
    backend: str,
    artifact_path: str | Path,
    name: str = "GEMM compare export",
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
    verify_local: bool = False,
    meta: dict[str, Any] | None = None,
    include_inputs: bool = False,
) -> int:
    """Rollout on the real-GPU machine: save quantized state + real output for replay elsewhere."""
    import os

    if seed is None:
        seed = int(time.time() * 1e6) ^ int.from_bytes(os.urandom(8), "little")
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
    dims_m = dims_m or [128, 256, 1024, 2048, 4096]
    dims_n = dims_n or [128, 256, 1024, 2048, 4096]
    dims_k = dims_k or [128, 256, 512, 1024]

    print(f"\n{'=' * 70}")
    print(name)
    print(f"{'=' * 70}")
    print(f"Mode: EXPORT -> {artifact_path}")
    print(f"Iterations: {num_iterations}")
    print(f"Distributions: {distributions}")
    print(f"Dims M: {dims_m} | N: {dims_n} | K: {dims_k}")
    print(f"verify_local (also run emul here): {verify_local}")
    print("=" * 70)

    cases: list[dict[str, Any]] = []
    local_mismatch_print = 0

    for i in range(num_iterations):
        M = random.choice(dims_m)
        N = random.choice(dims_n)
        K = random.choice(dims_k)
        da, db = random.choice(distributions), random.choice(distributions)

        print(f"\rExport {i + 1}/{num_iterations}: K={K:4}, A={da:10}, B={db:10} ... ", end="")
        sys.stdout.flush()

        try:
            row = run_test_case_export(
                i,
                M,
                N,
                K,
                da,
                db,
                quant_fn,
                real_fn,
                emul_fn,
                backend=backend,
                sample_count=sample_count,
                device=device,
                dtype=dtype,
                data_gen=data_gen,
                verify_local=verify_local,
                include_inputs=include_inputs,
            )
            cases.append(row)
            tag = "OK"
            lc = row.get("local_check")
            if lc is not None:
                tag = lc["status"]
                if lc["status"] == "MISMATCH" and local_mismatch_print < max_mismatch_print:
                    local_mismatch_print += 1
                    print(f"\n    >>> local emul MISMATCH (export machine) iter {i}:")
                    for d in lc["mismatch_details"]:
                        print(
                            f"      Pos {d['idx']}: Real={d['real']:.6f} | "
                            f"Model={d['model']:.6f} | Diff={d['diff']:.6f}"
                        )
            print(tag)
        except Exception as e:
            print(f"\nFATAL ERROR on export {i + 1}: {e}")
            traceback.print_exc()

    save_rollout_artifact(
        artifact_path,
        backend=backend,
        meta=meta or {},
        seed=seed,
        cases=cases,
    )
    print("\n" + "=" * 70)
    print(f"Saved {len(cases)} cases to {artifact_path}")
    print("=" * 70)
    if verify_local:
        bad = sum(1 for c in cases if (c.get("local_check") or {}).get("status") == "MISMATCH")
        if bad:
            print(f"Local emulation mismatches on export machine: {bad}/{len(cases)}")
            return 1
    return 0


def run_suite_import(
    emul_fn: GemmFn,
    *,
    backend: str,
    artifact_path: str | Path,
    name: str = "GEMM compare import",
    device: str = "cuda",
    sample_count: int = 5,
    max_mismatch_print: int = 20,
    metric_mode: str = "exact",
    print_iter_status: bool = True,
) -> int:
    """Replay saved rollout: compare artifact real output vs emulation on this machine."""
    data = load_rollout_artifact(artifact_path, map_location="cpu")
    art_backend = str(data["backend"]).lower()
    if art_backend != backend.lower():
        print(
            f"ERROR: artifact backend {art_backend!r} != CLI backend {backend.lower()!r}",
            file=sys.stderr,
        )
        return 2

    seed = data.get("seed")
    meta = data.get("meta", {})
    cases: list[dict[str, Any]] = list(data["cases"])

    print(f"\n{'=' * 70}")
    print(name)
    print(f"{'=' * 70}")
    print(f"Mode: IMPORT <- {artifact_path}")
    print(f"artifact seed = {seed}")
    print(f"artifact meta = {meta}")
    print(f"Cases: {len(cases)} | device = {device}")
    print("=" * 70)

    results: list[dict[str, Any]] = []
    mismatch_print_count = 0

    for idx, row in enumerate(cases):
        print(f"\rImport {idx + 1}/{len(cases)}: iter={row.get('iter')} ... ", end="")
        sys.stdout.flush()
        try:
            state = quant_state_from_dict(row["state"], backend, device=device)
            real_output = row["real_output"].to(device)
            modeling_res = emul_fn(state)
            if metric_mode == "exact":
                cmp = compare_tensors(real_output, modeling_res, sample_count=sample_count)
            elif metric_mode == "mse":
                cmp = compare_mse(real_output, modeling_res)
            else:
                raise ValueError(f"unknown metric_mode={metric_mode!r}")
            out = {
                "iter": row.get("iter", idx),
                "M": row["M"],
                "N": row["N"],
                "K": row["K"],
                "dist_a": row["dist_a"],
                "dist_b": row["dist_b"],
                **cmp,
            }
            results.append(out)
            if print_iter_status:
                print(out["status"])
            else:
                print("OK")
            if (
                metric_mode == "exact"
                and out["status"] == "MISMATCH"
                and mismatch_print_count < max_mismatch_print
            ):
                mismatch_print_count += 1
                print("    >>> Mismatch Details:")
                for d in out["mismatch_details"]:
                    print(
                        f"      Pos {d['idx']}: Real={d['real']:.6f} | "
                        f"Model={d['model']:.6f} | Diff={d['diff']:.6f}"
                    )
        except Exception as e:
            print(f"\nFATAL ERROR on import case {idx + 1}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"SUMMARY | {name}")
    print("=" * 70)
    if metric_mode == "mse":
        total_numel = sum(int(r.get("mse_numel", 0)) for r in results)
        total_mse_sum = sum(float(r.get("mse_sum", 0.0)) for r in results)
        mean_mse = total_mse_sum / max(total_numel, 1)
        per_gemm_scores = [float(r.get("per_gemm_rel_l2_error", 0.0)) for r in results]
        mean_per_gemm_score = sum(per_gemm_scores) / max(len(per_gemm_scores), 1)
        worst_iter_max = max((float(r.get("max_mse", 0.0)) for r in results), default=0.0)
        print(f"Total: {len(cases)}, Completed: {len(results)}")
        print(f"MSE mean: {mean_mse:.8e}")
        print(f"mean over GEMMs of sqrt(sum(error^2)) / sqrt(sum(real^2)): {mean_per_gemm_score:.8e}")
        print(f"MSE max (worst element): {worst_iter_max:.8e}")
        return 0

    mismatches = [r for r in results if r["status"] == "MISMATCH"]
    print(f"Total: {len(cases)}, Matches: {len(results) - len(mismatches)}, Mismatches: {len(mismatches)}")
    print("=" * 70)
    if len(mismatches) == 0:
        print("*** All tests: exact bitwise match ***")
        return 0
    print(f"Exact-match rate: {(len(results) - len(mismatches)) / max(len(results), 1) * 100:.2f}%")
    return 1
