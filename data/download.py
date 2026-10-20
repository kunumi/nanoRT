"""Download, validate, and install relation-aware data artifacts."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path

DATABASES = (
    "rel-amazon",
    "rel-hm",
    "rel-stack",
    "rel-avito",
    "rel-trial",
    "rel-f1",
)
REQUIRED_FILES = (
    "column_index.json",
    "nodes.rkyv",
    "offsets.rkyv",
    "p2f_adj.rkyv",
    "relation_text_idxs.json",
    "table_info.json",
)


def artifact_prefix(holdout: str) -> str:
    if holdout not in DATABASES:
        choices = ", ".join(DATABASES)
        raise ValueError(f"unknown holdout {holdout!r}; choose one of: {choices}")
    return f"ra-holdout-{holdout}"


def embedding_filename(model: str) -> str:
    return f"text_emb_{model}.bin"


def download_patterns(holdout: str, embedding_model: str) -> list[str]:
    prefix = artifact_prefix(holdout)
    names = (*REQUIRED_FILES, embedding_filename(embedding_model))
    roots = [
        *(f"{prefix}/train/{db}" for db in DATABASES if db != holdout),
        f"{prefix}/val/{holdout}",
    ]
    return [f"{root}/{name}" for root in roots for name in names]


def _read_relation_indices(path: Path) -> list[int]:
    try:
        values = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read relation metadata at {path}: {exc}") from exc
    if not isinstance(values, list) or any(
        not isinstance(value, int) or value < 0 for value in values
    ):
        raise ValueError(f"{path} must contain a JSON list of non-negative integers")
    if values != sorted(set(values)):
        raise ValueError(f"{path} must contain sorted, unique relation text indices")
    return values


def validate_database_artifact(
    path: Path, *, embedding_model: str, d_text: int
) -> None:
    required = (*REQUIRED_FILES, embedding_filename(embedding_model))
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(
            f"incomplete artifact at {path}; missing: {', '.join(missing)}"
        )

    relation_indices = _read_relation_indices(path / "relation_text_idxs.json")
    if not relation_indices:
        raise ValueError(f"relation-aware artifact at {path} has no relation indices")

    embedding_path = path / embedding_filename(embedding_model)
    bytes_per_row = 2 * d_text  # bfloat16
    size = embedding_path.stat().st_size
    if size == 0 or size % bytes_per_row:
        raise ValueError(
            f"{embedding_path} has {size} bytes; expected bfloat16 rows of width {d_text}"
        )
    rows = size // bytes_per_row
    if relation_indices[-1] >= rows:
        raise ValueError(
            f"relation index {relation_indices[-1]} exceeds {rows} embedding rows in "
            f"{embedding_path}"
        )


def validate_bundle(
    artifact_root: Path,
    *,
    holdout: str,
    embedding_model: str,
    d_text: int,
) -> dict[str, Path]:
    prefix = artifact_prefix(holdout)
    bundle_root = artifact_root / prefix
    train_root = bundle_root / "train"
    val_root = bundle_root / "val"
    expected_train = [db for db in DATABASES if db != holdout]

    installed: dict[str, Path] = {}
    for db in expected_train:
        path = train_root / db
        validate_database_artifact(path, embedding_model=embedding_model, d_text=d_text)
        installed[db] = path

    holdout_path = val_root / holdout
    validate_database_artifact(
        holdout_path, embedding_model=embedding_model, d_text=d_text
    )
    installed[holdout] = holdout_path

    observed_train = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    if observed_train != sorted(expected_train):
        raise ValueError(
            "training artifact violates holdout isolation: "
            f"expected {sorted(expected_train)}, found {observed_train}"
        )
    return installed


def install_links(paths: dict[str, Path], install_root: Path) -> None:
    install_root.mkdir(parents=True, exist_ok=True)
    for db, source in sorted(paths.items()):
        source = source.resolve()
        destination = install_root / db
        if destination.is_symlink():
            if destination.resolve() == source:
                continue
            destination.unlink()
        elif destination.exists():
            raise FileExistsError(
                f"refusing to replace existing path {destination}; move it first"
            )
        destination.symlink_to(source, target_is_directory=True)


def write_install_manifest(
    install_root: Path,
    *,
    repo_id: str,
    requested_revision: str,
    resolved_revision: str,
    holdout: str,
    embedding_model: str,
) -> None:
    manifest = {
        "schema_version": 1,
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "artifact_prefix": artifact_prefix(holdout),
        "holdout": holdout,
        "embedding_model": embedding_model,
        "training_databases": [db for db in DATABASES if db != holdout],
        "relation_aware": True,
    }
    (install_root / ".modded-nanort-artifacts.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _resolve_revision(repo_id: str, revision: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().repo_info(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
    )
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not resolve {repo_id}@{revision}")
    return info.sha


def _download(
    *,
    repo_id: str,
    revision: str,
    artifact_root: Path,
    patterns: Iterable[str],
) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=list(patterns),
        local_dir=artifact_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("NANORT_HF_REPO"),
        help="Hugging Face dataset id (or NANORT_HF_REPO)",
    )
    parser.add_argument(
        "--revision",
        default=os.environ.get("NANORT_HF_REVISION", "main"),
        help="branch, tag, or immutable commit (default: main)",
    )
    parser.add_argument(
        "--holdout",
        default=os.environ.get("NANORT_HOLDOUT_DB", "rel-avito"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(
            os.environ.get("NANORT_ARTIFACT_ROOT", "~/.cache/modded-nanort/artifacts")
        ).expanduser(),
    )
    parser.add_argument(
        "--install-root",
        type=Path,
        default=Path(
            os.environ.get("NANORT_PRE_ROOT", "~/.cache/modded-nanort/pre")
        ).expanduser(),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("NANORT_EMBEDDING_MODEL", "all-MiniLM-L12-v2"),
    )
    parser.add_argument(
        "--d-text", type=int, default=int(os.environ.get("NANORT_D_TEXT", "384"))
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="validate and install an already downloaded artifact without network access",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.repo_id:
        raise SystemExit("--repo-id or NANORT_HF_REPO is required")
    if args.d_text <= 0:
        raise SystemExit("--d-text must be positive")

    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.install_root = args.install_root.expanduser().resolve()
    patterns = download_patterns(args.holdout, args.embedding_model)

    if args.local_only:
        resolved_revision = args.revision
    else:
        resolved_revision = _resolve_revision(args.repo_id, args.revision)
        print(f"artifact: {args.repo_id}@{resolved_revision}", flush=True)
        _download(
            repo_id=args.repo_id,
            revision=resolved_revision,
            artifact_root=args.artifact_root,
            patterns=patterns,
        )

    paths = validate_bundle(
        args.artifact_root,
        holdout=args.holdout,
        embedding_model=args.embedding_model,
        d_text=args.d_text,
    )
    install_links(paths, args.install_root)
    write_install_manifest(
        args.install_root,
        repo_id=args.repo_id,
        requested_revision=args.revision,
        resolved_revision=resolved_revision,
        holdout=args.holdout,
        embedding_model=args.embedding_model,
    )

    print(f"installed relation-aware artifacts in {args.install_root}", flush=True)
    print(f"resolved revision: {resolved_revision}", flush=True)


if __name__ == "__main__":
    main()
