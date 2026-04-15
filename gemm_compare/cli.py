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
    parser.add_argument("--nvfp-out", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--nvfp-compare",
        choices=("real-emul", "pseudo-real"),
        default="real-emul",
        help="NVFP compare mode: real-emul (default) or pseudo-real (summary MSE).",
    )
    parser.add_argument("--mxfp-backend", default="cudnn", help="flashinfer mm_fp4 backend")
    parser.add_argument(
        "--mxfp-out",
        choices=("float16", "bfloat16"),
        default="float16",
        help="MXFP output dtype and random A/B dtype (float16 matches search/w3.py defaults)",
    )
    parser.add_argument(
        "--mxfp-compare",
        choices=("real-emul", "pseudo-real"),
        default="real-emul",
        help="MXFP compare mode: real-emul (default) or pseudo-real (summary normalized error).",
    )
    parser.add_argument("--group-size", type=int, default=32, help="MXFP block size")
    parser.add_argument(
        "--mode",
        choices=("local", "export", "import"),
        default="local",
        help="local: real vs emul here | export: save quant state + real output | import: load artifact, emul vs saved real",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Local path to rollout .pt (required for export; for import use this or --hf-repo)",
    )
    parser.add_argument(
        "--export-verify-local",
        action="store_true",
        help="During export, also run emulation on this machine (sanity check; slower)",
    )
    parser.add_argument(
        "--export-include-inputs",
        action="store_true",
        help="During export, include original A/B inputs in rollout state (needed for NVFP pseudo comparison on import).",
    )
    parser.add_argument(
        "--import-compare",
        choices=("emul", "pseudo"),
        default="emul",
        help="With --mode import: compare saved real output against emul (default) or pseudo path.",
    )
    parser.add_argument(
        "--hf-repo",
        default=None,
        metavar="REPO_ID",
        help="Hugging Face Hub repo id (e.g. user/gemm-rollouts). With export: upload after save. With import: download instead of --artifact.",
    )
    parser.add_argument(
        "--hf-path",
        default="gemm_compare_rollout.pt",
        help="Path of the rollout file inside the Hub repo (default: gemm_compare_rollout.pt)",
    )
    parser.add_argument(
        "--hf-repo-type",
        choices=("model", "dataset"),
        default="model",
        help="Hub repo type for upload/download (default: model)",
    )
    parser.add_argument("--hf-revision", default=None, help="Hub revision (branch / tag / commit)")
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HF API token (optional; otherwise HF_TOKEN env or cached huggingface-cli login)",
    )
    parser.add_argument(
        "--hf-create-repo",
        action="store_true",
        help="With export + --hf-repo: create the repo on the Hub if it does not exist",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="With --hf-create-repo: create the repo as private",
    )
    args = parser.parse_args()

    if args.mode == "export" and args.artifact is None:
        print("--artifact is required for --mode export (local file to write before optional Hub upload).", file=sys.stderr)
        return 2
    if args.mode == "import" and args.artifact is None and args.hf_repo is None:
        print("For --mode import, pass --artifact (local .pt) or --hf-repo (download from Hub).", file=sys.stderr)
        return 2

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA required for these backends.", file=sys.stderr)
        return 2

    from gemm_compare.runner import run_suite, run_suite_export, run_suite_import

    def import_artifact_path() -> Path:
        if args.artifact is not None:
            return args.artifact.expanduser().resolve()
        from gemm_compare.hf_hub_rollout import download_rollout_from_hub

        return download_rollout_from_hub(
            repo_id=args.hf_repo,
            path_in_repo=args.hf_path,
            repo_type=args.hf_repo_type,
            revision=args.hf_revision,
            token=args.hf_token,
        )

    def maybe_push_hf(local_path: Path) -> int:
        if not args.hf_repo:
            return 0
        from gemm_compare.hf_hub_rollout import upload_rollout_to_hub

        try:
            upload_rollout_to_hub(
                local_path,
                repo_id=args.hf_repo,
                path_in_repo=args.hf_path,
                repo_type=args.hf_repo_type,
                revision=args.hf_revision,
                token=args.hf_token,
                create_repo=args.hf_create_repo,
                private=args.hf_private,
            )
            rtype = "datasets" if args.hf_repo_type == "dataset" else "models"
            rev = args.hf_revision or "main"
            print(
                f"HF Hub: uploaded {args.hf_path!r} to https://huggingface.co/{rtype}/{args.hf_repo}/blob/{rev}/{args.hf_path}"
            )
            return 0
        except Exception as e:
            print(f"HF Hub upload failed: {e}", file=sys.stderr)
            return 3

    if args.backend == "nvfp":
        out_dtype = torch.float16 if args.nvfp_out == "float16" else torch.bfloat16
        dtype = torch.float16
        if args.mode == "import":
            if args.import_compare == "pseudo":
                from gemm_compare.backends.nvfp import build_nvfp_pseudo_fn

                compare_fn, meta = build_nvfp_pseudo_fn()
                name = f"NVFP import pseudo | {meta}"
                metric_mode = "mse"
                print_iter_status = False
            else:
                from gemm_compare.backends.nvfp import build_nvfp_emul_fn

                compare_fn, meta = build_nvfp_emul_fn(out_dtype=out_dtype)
                name = f"NVFP import | {meta}"
                metric_mode = "exact"
                print_iter_status = True
            ap = import_artifact_path()
            if args.hf_repo and args.artifact is None:
                print(f"HF Hub: using cached file {ap}")
            return run_suite_import(
                compare_fn,
                backend="nvfp",
                artifact_path=ap,
                name=name,
                device=args.device,
                metric_mode=metric_mode,
                print_iter_status=print_iter_status,
            )
        from gemm_compare.backends.nvfp import build_nvfp_fns

        if args.nvfp_compare == "pseudo-real":
            quant_fn, real_fn, emul_fn, pseudo_fn, meta = build_nvfp_fns(
                out_dtype=out_dtype, return_pseudo=True
            )
            name = f"NVFP pseudo vs real | {meta}"
            return run_suite(
                quant_fn,
                real_fn,
                pseudo_fn,
                name=name,
                num_iterations=args.iterations,
                device=args.device,
                dtype=dtype,
                seed=args.seed,
                metric_mode="mse",
                print_iter_status=False,
            )

        quant_fn, real_fn, emul_fn, meta = build_nvfp_fns(out_dtype=out_dtype)
        name = f"NVFP | {meta}"
        if args.mode == "export":
            ret = run_suite_export(
                quant_fn,
                real_fn,
                emul_fn,
                backend="nvfp",
                artifact_path=args.artifact,
                name=name,
                num_iterations=args.iterations,
                device=args.device,
                dtype=dtype,
                seed=args.seed,
                verify_local=args.export_verify_local,
                meta=meta,
                include_inputs=args.export_include_inputs,
            )
            up = maybe_push_hf(args.artifact.expanduser().resolve())
            return up if up != 0 else ret
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

    mxfp_out = torch.float16 if args.mxfp_out == "float16" else torch.bfloat16
    if args.mode == "import":
        if args.import_compare == "pseudo":
            from gemm_compare.backends.mxfp import build_mxfp_pseudo_fn

            compare_fn, meta = build_mxfp_pseudo_fn()
            name = f"MXFP import pseudo | {meta}"
            metric_mode = "mse"
            print_iter_status = False
        else:
            from gemm_compare.backends.mxfp import build_mxfp_emul_fn

            compare_fn, meta = build_mxfp_emul_fn(
                group_size=args.group_size,
                out_dtype=mxfp_out,
            )
            name = f"MXFP import | {meta}"
            metric_mode = "exact"
            print_iter_status = True
        ap = import_artifact_path()
        if args.hf_repo and args.artifact is None:
            print(f"HF Hub: using cached file {ap}")
        return run_suite_import(
            compare_fn,
            backend="mxfp",
            artifact_path=ap,
            name=name,
            device=args.device,
            metric_mode=metric_mode,
            print_iter_status=print_iter_status,
        )

    from gemm_compare.backends.mxfp import build_mxfp_fns

    if args.mxfp_compare == "pseudo-real":
        quant_fn, real_fn, emul_fn, pseudo_fn, meta = build_mxfp_fns(
            group_size=args.group_size,
            out_dtype=mxfp_out,
            mm_backend=args.mxfp_backend,
            return_pseudo=True,
        )
        name = f"MXFP pseudo vs real | {meta}"
        return run_suite(
            quant_fn,
            real_fn,
            pseudo_fn,
            name=name,
            num_iterations=args.iterations,
            device=args.device,
            dtype=mxfp_out,
            seed=args.seed,
            metric_mode="mse",
            print_iter_status=False,
        )

    quant_fn, real_fn, emul_fn, meta = build_mxfp_fns(
        group_size=args.group_size,
        out_dtype=mxfp_out,
        mm_backend=args.mxfp_backend,
    )
    name = f"MXFP | {meta}"
    if args.mode == "export":
        ret = run_suite_export(
            quant_fn,
            real_fn,
            emul_fn,
            backend="mxfp",
            artifact_path=args.artifact,
            name=name,
            num_iterations=args.iterations,
            device=args.device,
            dtype=mxfp_out,
            seed=args.seed,
            verify_local=args.export_verify_local,
            meta=meta,
            include_inputs=args.export_include_inputs,
        )
        up = maybe_push_hf(args.artifact.expanduser().resolve())
        return up if up != 0 else ret
    return run_suite(
        quant_fn,
        real_fn,
        emul_fn,
        name=name,
        num_iterations=args.iterations,
        device=args.device,
        dtype=mxfp_out,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
