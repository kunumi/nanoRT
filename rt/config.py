"""Single source of truth for training configuration.

All knobs live in ``TrainConfig``. The only runtime entry point (``scripts/
nanort_train.py``) builds one via ``TrainConfig.from_env()`` and hands it to
``rt.main.main`` — there is no command-line flag parsing anywhere.

This module deliberately has **no torch import** so the config surface can be
inspected and unit-tested without a GPU environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

OFFICIAL_TARGET_AUROC = 0.81621


# ── env parsing helpers ────────────────────────────────────────────────────
def _str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().strip("\"'")


def _int(name: str, default: int) -> int:
    return int(_str(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_str(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _str(name, "1" if default else "0").lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


def _opt_int(name: str, default, *, disabled_below: int | None = None):
    """Read an int env var; return None when unset or (optionally) when the
    value is a sentinel below ``disabled_below`` (e.g. sigreg_layer=-1, or
    lr_total_steps=0 → 'use max_steps')."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    v = int(raw.strip().strip("\"'"))
    if disabled_below is not None and v < disabled_below:
        return None
    return v


def _opt_float(name: str, default, *, zero_is_none: bool = False):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    v = float(raw.strip().strip("\"'"))
    if zero_is_none and v == 0.0:
        return None
    return v


def _simplicial_layers(name: str):
    """'all' → None (every block); 'none' → []; '0,4,8' → [0,4,8]. Tolerant of
    list-like values such as '[0, 4]' (quotes and spaces allowed)."""
    raw = os.environ.get(name, "all")
    clean = raw.strip().strip("[]()\"'").strip().lower()
    if clean in ("all", ""):
        return None
    if clean == "none":
        return []
    return [int(tok.strip("\"' ")) for tok in clean.split(",") if tok.strip("\"' ")]


@dataclass
class TrainConfig:
    """Every training knob. Defaults reproduce the competition run; ``from_env``
    overlays the ``NANORT_*`` environment variables."""

    # ── run identity ────────────────────────────────────────────────────────
    holdout_db: str = "rel-avito"
    seed: int = 0
    project: str = "modded-nanort"

    # ── data ────────────────────────────────────────────────────────────────
    batch_size: int = 256
    num_workers: int = 8
    max_bfs_width: int = 256
    seq_len: int = 1024
    embedding_model: str = "all-MiniLM-L12-v2"
    d_text: int = 384
    # Resolved at runtime from holdout_db (not env-configurable).
    train_tasks: list = field(default_factory=list)
    val_tasks: list = field(default_factory=list)

    # ── optimization ────────────────────────────────────────────────────────
    optimizer: str = "muon"  # "adamw" | "muon"
    lr: float = 1e-3  # AdamW group LR
    muon_lr: float = 5e-3  # Muon group LR (ignored when optimizer="adamw")
    muon_warmup_steps: int = 3000
    wd: float = 0.1
    lr_schedule: bool = True
    max_grad_norm: float = 1.0
    max_steps: int = 1_000_000
    lr_total_steps: int | None = None  # OneCycle horizon; None → max_steps

    # ── model ───────────────────────────────────────────────────────────────
    num_blocks: int = 12
    d_model: int = 256
    num_heads: int = 8
    d_ff: int = 1024
    simplicial_layers: list | None = None  # None = all blocks; [] = disabled
    update_relations: bool = False  # Record: static text-relation stream
    qk_norm: bool = True
    simplicial_layerscale_init: float = 1.0

    # ── SIGReg (geometry regularization; sigreg_layer=None disables it) ──────
    sigreg_layer: int | None = 0
    sigreg_lambda_row: float = 0.0
    sigreg_lambda_col: float = 1e-3
    sigreg_lr: float | None = None
    sigreg_n_proj: int = 64
    sigreg_proj_hidden_mult: int = 4
    sigreg_proj_out_div: int = 4
    sigreg_proj_hidden_layers: int = 1
    sigreg_knots: int = 17
    sigreg_t_max: float = 3.0
    sigreg_min_group: int = 8
    sigreg_col_global_pool: bool = True

    # ── evaluation / budget / checkpoint ────────────────────────────────────
    val_eval_freq: int = 100
    eval_at_step1: bool = False
    # Zero-gradient baseline: evaluate the model exactly as initialised, before
    # any optimizer step. Separates "the init already scores this" from "training
    # got here" — the two are indistinguishable when the first eval is at step 1.
    eval_at_step0: bool = True
    # Dense early evals: up to `val_eval_dense_until`, evaluate every
    # `val_eval_dense_freq` steps instead of every `val_eval_freq`. The zero-shot
    # AUROC peaks within the first few dozen steps, which a freq of 100 misses
    # entirely. 0 disables (plain val_eval_freq everywhere).
    val_eval_dense_until: int = 0
    val_eval_dense_freq: int = 5
    time_budget_minutes: float | None = 45.0
    save_ckpt_dir: str | None = None
    compile_: bool = True

    @classmethod
    def from_env(cls) -> TrainConfig:
        """Build a config from the ``NANORT_*`` environment. Unset vars fall
        back to the dataclass defaults above."""
        cfg = cls(
            holdout_db=_str("NANORT_HOLDOUT_DB", cls.holdout_db),
            seed=_int("NANORT_SEED", cls.seed),
            project=_str("WANDB_PROJECT", cls.project),
            batch_size=_int("NANORT_BATCH_SIZE", cls.batch_size),
            num_workers=_int("NANORT_NUM_WORKERS", cls.num_workers),
            max_bfs_width=_int("NANORT_MAX_BFS_WIDTH", cls.max_bfs_width),
            seq_len=_int("NANORT_SEQ_LEN", cls.seq_len),
            embedding_model=_str("NANORT_EMBEDDING_MODEL", cls.embedding_model),
            d_text=_int("NANORT_D_TEXT", cls.d_text),
            optimizer=_str("NANORT_OPTIMIZER", cls.optimizer),
            lr=_float("NANORT_LR", cls.lr),
            muon_lr=_float("NANORT_MUON_LR", cls.muon_lr),
            muon_warmup_steps=_int("NANORT_MUON_WARMUP_STEPS", cls.muon_warmup_steps),
            wd=_float("NANORT_WEIGHT_DECAY", cls.wd),
            lr_schedule=_bool("NANORT_LR_SCHEDULE", cls.lr_schedule),
            max_grad_norm=_float("NANORT_MAX_GRAD_NORM", cls.max_grad_norm),
            max_steps=_int("NANORT_MAX_STEPS", cls.max_steps),
            lr_total_steps=_opt_int("NANORT_LR_TOTAL_STEPS", None, disabled_below=1),
            num_blocks=_int("NANORT_NUM_BLOCKS", cls.num_blocks),
            d_model=_int("NANORT_D_MODEL", cls.d_model),
            num_heads=_int("NANORT_NUM_HEADS", cls.num_heads),
            d_ff=_int("NANORT_D_FF", cls.d_ff),
            simplicial_layers=_simplicial_layers("NANORT_SIMPLICIAL_LAYERS"),
            update_relations=_bool("NANORT_UPDATE_RELATIONS", cls.update_relations),
            qk_norm=_bool("NANORT_QK_NORM", cls.qk_norm),
            simplicial_layerscale_init=_float(
                "NANORT_SIMPLICIAL_LAYERSCALE_INIT", cls.simplicial_layerscale_init
            ),
            sigreg_layer=_opt_int(
                "NANORT_SIGREG_LAYER", cls.sigreg_layer, disabled_below=0
            ),
            sigreg_lambda_row=_float("NANORT_SIGREG_LAMBDA_ROW", cls.sigreg_lambda_row),
            sigreg_lambda_col=_float("NANORT_SIGREG_LAMBDA_COL", cls.sigreg_lambda_col),
            sigreg_lr=_opt_float("NANORT_SIGREG_LR", None, zero_is_none=True),
            sigreg_n_proj=_int("NANORT_SIGREG_N_PROJ", cls.sigreg_n_proj),
            sigreg_proj_hidden_mult=_int(
                "NANORT_SIGREG_PROJ_HIDDEN_MULT", cls.sigreg_proj_hidden_mult
            ),
            sigreg_proj_out_div=_int(
                "NANORT_SIGREG_PROJ_OUT_DIV", cls.sigreg_proj_out_div
            ),
            sigreg_proj_hidden_layers=_int(
                "NANORT_SIGREG_PROJ_HIDDEN_LAYERS", cls.sigreg_proj_hidden_layers
            ),
            sigreg_knots=_int("NANORT_SIGREG_KNOTS", cls.sigreg_knots),
            sigreg_t_max=_float("NANORT_SIGREG_T_MAX", cls.sigreg_t_max),
            sigreg_min_group=_int("NANORT_SIGREG_MIN_GROUP", cls.sigreg_min_group),
            sigreg_col_global_pool=_bool(
                "NANORT_SIGREG_COL_POOL", cls.sigreg_col_global_pool
            ),
            val_eval_freq=_int("NANORT_VAL_EVAL_FREQ", cls.val_eval_freq),
            eval_at_step1=_bool("NANORT_EVAL_AT_STEP1", cls.eval_at_step1),
            eval_at_step0=_bool("NANORT_EVAL_AT_STEP0", cls.eval_at_step0),
            val_eval_dense_until=_int(
                "NANORT_VAL_EVAL_DENSE_UNTIL", cls.val_eval_dense_until
            ),
            val_eval_dense_freq=_int(
                "NANORT_VAL_EVAL_DENSE_FREQ", cls.val_eval_dense_freq
            ),
            time_budget_minutes=_opt_float(
                "NANORT_TIME_BUDGET_MINUTES",
                cls.time_budget_minutes,
                zero_is_none=True,
            ),
            save_ckpt_dir=os.environ.get("NANORT_SAVE_CKPT_DIR") or None,
            compile_=_bool("NANORT_COMPILE", cls.compile_),
        )
        return cfg.validate()

    def validate(self) -> TrainConfig:
        """Fail fast on invalid or scientifically ambiguous configurations."""
        if self.optimizer not in {"adamw", "muon"}:
            raise ValueError(
                f"optimizer must be 'adamw' or 'muon', got {self.optimizer!r}"
            )
        if self.batch_size <= 0 or self.seq_len <= 0 or self.max_bfs_width <= 0:
            raise ValueError("batch_size, seq_len and max_bfs_width must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if (
            min(self.num_blocks, self.d_model, self.d_text, self.num_heads, self.d_ff)
            <= 0
        ):
            raise ValueError("all model dimensions and num_blocks must be positive")
        if self.d_model % self.num_heads:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by num_heads={self.num_heads}"
            )
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.lr_total_steps is not None and self.lr_total_steps < self.max_steps:
            raise ValueError(
                "lr_total_steps must be >= max_steps to avoid scheduler overrun"
            )
        if self.lr <= 0 or self.muon_lr <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning rates and max_grad_norm must be positive")
        if self.wd < 0:
            raise ValueError("weight decay must be non-negative")
        if self.muon_warmup_steps < 0:
            raise ValueError("muon_warmup_steps must be non-negative")
        if self.simplicial_layerscale_init < 0:
            raise ValueError("simplicial_layerscale_init must be non-negative")
        if self.simplicial_layers is not None:
            invalid = [
                layer
                for layer in self.simplicial_layers
                if layer < 0 or layer >= self.num_blocks
            ]
            if invalid:
                raise ValueError(
                    f"simplicial layer indices out of range [0, {self.num_blocks}): {invalid}"
                )
        if (
            self.sigreg_layer is not None
            and not 0 <= self.sigreg_layer < self.num_blocks
        ):
            raise ValueError(
                f"sigreg_layer={self.sigreg_layer} out of range [0, {self.num_blocks})"
            )
        if self.sigreg_lambda_row < 0 or self.sigreg_lambda_col < 0:
            raise ValueError("SIGReg lambdas must be non-negative")
        if self.sigreg_lambda_row + self.sigreg_lambda_col >= 1:
            raise ValueError("SIGReg lambdas must sum to less than 1")
        if (
            self.sigreg_layer is None
            and self.sigreg_lambda_row + self.sigreg_lambda_col > 0
        ):
            raise ValueError("positive SIGReg lambda requires sigreg_layer")
        if self.sigreg_lr is not None and self.sigreg_lr <= 0:
            raise ValueError("sigreg_lr must be positive when provided")
        if (
            min(
                self.sigreg_n_proj,
                self.sigreg_proj_hidden_mult,
                self.sigreg_proj_out_div,
                self.sigreg_proj_hidden_layers,
                self.sigreg_min_group,
            )
            <= 0
        ):
            raise ValueError(
                "SIGReg projection and grouping parameters must be positive"
            )
        if self.sigreg_knots < 2 or self.sigreg_t_max <= 0:
            raise ValueError("SIGReg quadrature requires knots >= 2 and t_max > 0")
        if self.val_eval_freq <= 0:
            raise ValueError("val_eval_freq must be positive when provided")
        if self.val_eval_dense_until < 0 or self.val_eval_dense_freq <= 0:
            raise ValueError(
                "dense eval window must be non-negative with positive frequency"
            )
        if self.time_budget_minutes is not None and self.time_budget_minutes <= 0:
            raise ValueError("time_budget_minutes must be positive when provided")
        return self

    def summary(self) -> str:
        """One-line human-readable dump of the non-task fields (for logs)."""
        skip = {"train_tasks", "val_tasks"}
        return "  ".join(
            f"{f.name}={getattr(self, f.name)}"
            for f in fields(self)
            if f.name not in skip
        )
