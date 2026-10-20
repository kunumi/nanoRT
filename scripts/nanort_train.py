"""modded-nanoRT training entry point.

All configuration lives in ``rt.config.TrainConfig`` and is populated from the
``NANORT_*`` environment variables. There is no command-line flag parsing; set
environment variables to change the record recipe.

Run:
    NANORT_HOLDOUT_DB=rel-avito pixi run python scripts/nanort_train.py
"""

import json
import os
from pathlib import Path

from rt.config import TrainConfig
from rt.tasks import autocomplete_clf_tasks, forecast_clf_tasks

PRE_ROOT = Path(
    os.environ.get("NANORT_PRE_ROOT", Path.home() / ".cache" / "modded-nanort" / "pre")
).expanduser()
INSTALL_MANIFEST = PRE_ROOT / ".modded-nanort-artifacts.json"

HOLDOUT_CANDIDATES = [
    "rel-amazon",
    "rel-hm",
    "rel-stack",
    "rel-avito",
    "rel-trial",
    "rel-f1",
]

# Competition protocol — do NOT modify. rel-event is excluded from the pool.
ALL_CLF_TASKS = [
    t for t in forecast_clf_tasks + autocomplete_clf_tasks if t[0] != "rel-event"
]


def clf_tasks_for_db(db_name: str):
    return [t for t in ALL_CLF_TASKS if t[0] == db_name]


def load_artifact_provenance() -> None:
    """Restore the immutable HF identity written by ``data/download.py``."""
    if not INSTALL_MANIFEST.is_file():
        return
    with INSTALL_MANIFEST.open() as file:
        manifest = json.load(file)
    mappings = {
        "NANORT_ARTIFACT_REPO": "repo_id",
        "NANORT_ARTIFACT_REVISION": "resolved_revision",
        "NANORT_ARTIFACT_PREFIX": "artifact_prefix",
    }
    for environment_name, manifest_name in mappings.items():
        value = manifest.get(manifest_name)
        if value:
            os.environ.setdefault(environment_name, str(value))


def resolve_tasks(cfg: TrainConfig) -> TrainConfig:
    """Fill cfg.train_tasks / cfg.val_tasks from cfg.holdout_db and validate that
    the required preprocessed artifacts are present. Returns the same cfg."""
    cfg.validate()
    if cfg.holdout_db not in HOLDOUT_CANDIDATES:
        raise ValueError(
            f"Unknown holdout {cfg.holdout_db!r}; expected one of {HOLDOUT_CANDIDATES}"
        )

    train_dbs = [db for db in HOLDOUT_CANDIDATES if db != cfg.holdout_db]
    train_artifact_paths = set()
    for db in train_dbs:
        db_path = PRE_ROOT / db
        if not db_path.is_dir():
            raise FileNotFoundError(
                f"Train artifact not found at {db_path}.\n"
                "Install it first with data/download.py."
            )
        train_artifact_paths.add(db_path.resolve())
    if len(train_artifact_paths) != len(train_dbs):
        raise RuntimeError(
            "Holdout isolation failed: two training database names resolve to "
            "the same physical artifact"
        )

    cfg.train_tasks = [
        (db, table, target, drop)
        for db, table, target, drop in ALL_CLF_TASKS
        if db != cfg.holdout_db
    ]

    observed_train_dbs = {task[0] for task in cfg.train_tasks}
    expected_train_dbs = set(train_dbs)
    if cfg.holdout_db in observed_train_dbs or observed_train_dbs != expected_train_dbs:
        raise RuntimeError(
            "Holdout isolation failed: "
            f"holdout={cfg.holdout_db}, train_dbs={sorted(observed_train_dbs)}, "
            f"expected={sorted(expected_train_dbs)}"
        )

    val_artifact_path = PRE_ROOT / cfg.holdout_db
    if not val_artifact_path.is_dir():
        raise FileNotFoundError(
            f"Val artifact not found at {val_artifact_path}.\n"
            "Install it first with data/download.py."
        )
    if val_artifact_path.resolve() in train_artifact_paths:
        raise RuntimeError(
            "Holdout isolation failed: the validation artifact resolves to a "
            "training artifact path"
        )
    with (val_artifact_path / "table_info.json").open() as f:
        val_table_info = json.load(f)

    cfg.val_tasks = [
        (cfg.holdout_db, table, target, "Val", drop)
        for _, table, target, drop in clf_tasks_for_db(cfg.holdout_db)
        if f"{table}:Val" in val_table_info
    ]
    if not cfg.val_tasks:
        raise RuntimeError(f"No val tasks found in {val_artifact_path}/table_info.json")
    if any(task[0] != cfg.holdout_db for task in cfg.val_tasks):
        raise RuntimeError("Validation tasks must all belong to the held-out database")
    print(
        f"Val tasks ({len(cfg.val_tasks)}): {[t[1] for t in cfg.val_tasks]}", flush=True
    )
    return cfg


def default_ckpt_dir(cfg: TrainConfig) -> str:
    tag = cfg.optimizer
    if cfg.optimizer == "muon":
        tag += f"-lr{cfg.muon_lr}"
    if (
        cfg.sigreg_layer is not None
        and (cfg.sigreg_lambda_row + cfg.sigreg_lambda_col) > 0
    ):
        tag += f"-sig-l{cfg.sigreg_layer}-c{cfg.sigreg_lambda_col}"
    if cfg.simplicial_layers != []:
        layers = (
            "all"
            if cfg.simplicial_layers is None
            else "-".join(str(layer) for layer in cfg.simplicial_layers)
        )
        tag += f"-simp-{layers}"
    if not cfg.update_relations:
        tag += "-staticrel"
    return f"ckpts/nanort-{cfg.holdout_db}-{tag}-s{cfg.seed}"


def run(cfg: TrainConfig) -> None:
    # Keep the protocol/task resolver importable in CPU-only environments.
    from rt.main import main

    resolve_tasks(cfg)
    if cfg.save_ckpt_dir is None:
        cfg.save_ckpt_dir = default_ckpt_dir(cfg)
    print(f"[config] {cfg.summary()}", flush=True)
    main(cfg)


if __name__ == "__main__":
    load_artifact_provenance()
    cfg = TrainConfig.from_env()
    try:
        run(cfg)
    except Exception:
        # Preserve the traceback in W&B when a run is active and always flush
        # it to stdout for local or batch execution.
        import traceback

        tb = traceback.format_exc()
        print(tb, flush=True)
        try:
            import wandb

            if wandb.run is not None:
                wandb.run.summary["crash_traceback"] = tb
                wandb.finish(exit_code=1)
        except Exception as logging_error:  # noqa: BLE001 - preserve original crash
            print(f"Failed to persist crash metadata: {logging_error}", flush=True)
        raise
