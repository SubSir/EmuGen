"""Save / load quantized GEMM state + real GPU output for cross-machine emulation replay."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

ARTIFACT_VERSION = 1


def _torch_load(path: str | Path, *, map_location: str | torch.device | None = None) -> Any:
    kwargs: dict[str, Any] = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def quant_state_to_cpu_dict(state: Any, backend: str) -> dict[str, Any]:
    b = backend.lower()
    if b == "nvfp":
        from gemm_compare.backends.nvfp import nvfp_quant_state_to_cpu_dict

        return nvfp_quant_state_to_cpu_dict(state)
    if b == "mxfp":
        from gemm_compare.backends.mxfp import mxfp_quant_state_to_cpu_dict

        return mxfp_quant_state_to_cpu_dict(state)
    raise ValueError(f"unknown backend {backend!r} for rollout")


def quant_state_from_dict(d: dict[str, Any], backend: str, *, device: str | torch.device) -> Any:
    b = backend.lower()
    if b == "nvfp":
        from gemm_compare.backends.nvfp import nvfp_quant_state_from_dict

        return nvfp_quant_state_from_dict(d, device=device)
    if b == "mxfp":
        from gemm_compare.backends.mxfp import mxfp_quant_state_from_dict

        return mxfp_quant_state_from_dict(d, device=device)
    raise ValueError(f"unknown backend {backend!r} for rollout")


def save_rollout_artifact(
    path: str | Path,
    *,
    backend: str,
    meta: dict[str, Any],
    seed: int,
    cases: list[dict[str, Any]],
) -> None:
    payload = {
        "gemm_compare_rollout_version": ARTIFACT_VERSION,
        "backend": backend.lower(),
        "meta": meta,
        "seed": seed,
        "cases": cases,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_rollout_artifact(path: str | Path, *, map_location: str | torch.device | None = "cpu") -> dict[str, Any]:
    data = _torch_load(path, map_location=map_location)
    if not isinstance(data, dict):
        raise ValueError(f"artifact is not a dict: {path!r}")
    ver = data.get("gemm_compare_rollout_version")
    if ver != ARTIFACT_VERSION:
        raise ValueError(
            f"unsupported rollout artifact version {ver!r} (this code expects {ARTIFACT_VERSION})"
        )
    if "backend" not in data or "cases" not in data:
        raise ValueError(f"malformed rollout artifact: {path!r}")
    return data
