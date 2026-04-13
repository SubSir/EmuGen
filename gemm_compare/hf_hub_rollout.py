"""Push / fetch rollout artifacts on the Hugging Face Hub (optional ``huggingface_hub``)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

RepoType = Literal["model", "dataset"]


def _require_hf_hub():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Hugging Face rollout requires the `huggingface_hub` package. "
            "Install with: pip install huggingface_hub"
        ) from e


def upload_rollout_to_hub(
    local_path: str | Path,
    *,
    repo_id: str,
    path_in_repo: str,
    repo_type: RepoType = "model",
    revision: str | None = None,
    token: str | None = None,
    create_repo: bool = False,
    private: bool = False,
    commit_message: str | None = None,
) -> None:
    """Upload a local ``.pt`` rollout to ``repo_id`` at ``path_in_repo``."""
    _require_hf_hub()
    from huggingface_hub import HfApi

    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(f"rollout file not found: {local_path}")

    api = HfApi(token=token)
    if create_repo:
        api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        commit_message=commit_message or f"Upload gemm_compare rollout {path_in_repo}",
    )


def download_rollout_from_hub(
    *,
    repo_id: str,
    path_in_repo: str,
    repo_type: RepoType = "model",
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download a rollout from the Hub; returns a path in the local HF cache."""
    _require_hf_hub()
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )
    return Path(p)
