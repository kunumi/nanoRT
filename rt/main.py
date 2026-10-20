import os
import platform
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from sklearn.metrics import r2_score, roc_auc_score
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grads_with_norm_, get_total_norm
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rt.config import OFFICIAL_TARGET_AUROC, TrainConfig
from rt.data import RelationalDataset, get_column_index
from rt.model import RelationalTransformer
from rt.optim import Muon, build_muon_param_groups
from rt.val_metrics import decompose_val_auroc, eval_is_due, safe_auroc


def _git_value(*args: str) -> str | None:
    """Return a small Git provenance field, or ``None`` outside a checkout."""
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _reproducibility_metadata(device: str) -> dict[str, str | float | bool | None]:
    commit = _git_value("rev-parse", "HEAD")
    status = _git_value("status", "--porcelain")
    return {
        "official_target_auroc": OFFICIAL_TARGET_AUROC,
        "artifact_repo": os.environ.get("NANORT_ARTIFACT_REPO"),
        "artifact_revision": os.environ.get("NANORT_ARTIFACT_REVISION"),
        "artifact_prefix": os.environ.get("NANORT_ARTIFACT_PREFIX"),
        "git_commit": commit,
        "git_dirty": None if commit is None else status is not None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
    }


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the raw module behind DDP and/or ``torch.compile`` wrappers."""
    while True:
        if isinstance(model, DDP):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


def _canonical_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Checkpoint without wrapper-specific ``module.``/``_orig_mod.`` prefixes."""
    return _unwrap_model(model).state_dict()


def _load_model_state(model: torch.nn.Module, path: Path) -> None:
    """Load current or historical raw/compiled/DDP nanoRT checkpoints."""
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected a state_dict mapping in {path}, got {type(state_dict)}"
        )

    canonical = {}
    for name, value in state_dict.items():
        clean = name
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if clean.startswith(prefix):
                    clean = clean[len(prefix) :]
                    changed = True
        canonical[clean] = value
    model.load_state_dict(canonical)


# NanoRT: 45-minute wall-clock training timer ──────────────────────────────────
class TrainTimer:
    """Count historical timed wall-clock after the first optimizer update."""

    def __init__(self, budget_minutes: float):
        self.budget_secs = budget_minutes * 60.0
        self._t0: float | None = None

    def start(self) -> None:
        if self._t0 is None:
            self._t0 = time.perf_counter()

    def elapsed(self) -> float:
        if self._t0 is None:
            return 0.0
        return time.perf_counter() - self._t0

    def over_budget(self) -> bool:
        return self.elapsed() > self.budget_secs


# NanoRT: zero-shot AUROC logging callback ─────────────────────────────────────
class ValAUROCCallback:
    """Computes zero-shot AUROC on the fixed val artifact at regular intervals.

    Tracks the wall-clock time at which the AUROC target is first reached
    (logged as train/time_to_target; -1 if not reached within budget).
    Only runs on rank 0 — not DDP-safe, fine for the 1×H200 setup.
    """

    TARGET_AUROC = OFFICIAL_TARGET_AUROC

    def __init__(
        self,
        val_tasks,
        batch_size,
        seq_len,
        max_bfs_width,
        embedding_model,
        d_text,
        num_workers,
        device,
        eval_every: int,
        timer: TrainTimer | None = None,
        eval_at_step1: bool = True,
        dense_until: int = 0,
        dense_freq: int = 5,
        save_ckpt_dir: str | None = None,
    ):
        val_dataset = RelationalDataset(
            tasks=val_tasks,
            batch_size=batch_size,
            seq_len=seq_len,
            rank=0,
            world_size=1,
            max_bfs_width=max_bfs_width,
            embedding_model=embedding_model,
            d_text=d_text,
            seed=0,
        )
        val_dataset.sampler.shuffle_py(0)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
            in_order=True,
        )
        # Target-cell column index -> task name, for the per-task AUROC split.
        # Each val sequence has exactly one target cell, and its col_name_idxs
        # entry is the text index of the target column — the same index
        # get_column_index resolves — so the batch already carries the task id.
        task_pairs = [
            (
                get_column_index(target_column, table_name, db_name),
                f"{table_name}/{target_column}",
            )
            for db_name, table_name, target_column, _split, _drop in val_tasks
        ]
        self.task_names = dict(task_pairs)
        if len(self.task_names) != len(task_pairs):
            raise ValueError(f"Validation task IDs are not unique: {task_pairs}")
        self.device = device
        self.eval_every = eval_every
        self.timer = timer
        self._time_to_target: float | None = None  # seconds; set once
        # Skip the step-1 eval when False: on the static-relation route the
        # near-init auroc is high (~0.87) and would trigger time_to_target at
        # step 1, which is not a trained-model crossing. Skipping it makes the
        # metric reflect a real checkpoint.
        self.eval_at_step1 = eval_at_step1
        # Evaluate every `dense_freq` steps up to `dense_until`, then fall back to
        # `eval_every`. The zero-shot auroc peaks within the first few dozen steps,
        # which the default eval_every=100 steps straight over.
        self.dense_until = dense_until
        self.dense_freq = max(1, dense_freq)
        # Save the best-by-zero-shot-auroc checkpoint. The auroc peaks early and
        # decays, so the budget-end checkpoint is NOT the best one to keep.
        self.save_ckpt_dir = save_ckpt_dir
        self.best_auroc = -float("inf")
        self.eval_seconds_total = 0.0

    def _due(self, step: int) -> bool:
        return eval_is_due(
            step,
            eval_every=self.eval_every,
            eval_at_step1=self.eval_at_step1,
            dense_until=self.dense_until,
            dense_freq=self.dense_freq,
        )

    def step(self, step: int, net: torch.nn.Module) -> None:
        if not self._due(step):
            return

        eval_started = time.perf_counter()
        net.eval()
        preds, labels, task_ids, weighted_losses = [], [], [], []
        stats_accum = {}
        stats_count = 0
        with torch.inference_mode():
            for batch in self.val_loader:
                true_batch_size = batch.pop("true_batch_size")
                for k in batch:
                    batch[k] = batch[k].to(self.device, non_blocking=True)
                batch["masks"][true_batch_size:, :] = False
                batch["is_targets"][true_batch_size:, :] = False
                batch["is_padding"][true_batch_size:, :] = True

                # need_stats defaults to True, which makes the model compute ~40
                # diagnostics per batch — including .unique() calls — and
                # _accumulate_stats then forces a GPU->CPU sync for every one of
                # them. Take a deterministic first-batch probe and label it as
                # such; official metrics still use the complete validation set.
                want_stats = stats_count == 0
                loss, yhat_dict, _stats = net(batch, need_stats=want_stats)
                if want_stats:
                    _accumulate_stats(stats_accum, _stats)
                    stats_count += 1
                yhat = yhat_dict["boolean"][batch["is_targets"]]
                y = batch["boolean_values"][batch["is_targets"]].flatten()

                weighted_losses.append((loss.item(), y.numel()))
                preds.append(yhat.flatten())
                labels.append(y)
                task_ids.append(batch["col_name_idxs"][batch["is_targets"]])

        preds_np = torch.cat(preds).float().cpu().numpy()
        labels_np = torch.cat(labels).float().cpu().numpy()
        labels_bin = [int(x > 0) for x in labels_np]
        task_ids_np = torch.cat(task_ids).cpu().numpy()

        auroc = safe_auroc(labels_bin, preds_np)
        n_loss_examples = sum(n for _, n in weighted_losses)
        mtp_loss = sum(value * n for value, n in weighted_losses) / max(
            n_loss_examples, 1
        )

        # auroc_zero_shot pools all val tasks into one number and is kept as-is:
        # it is what the record history and time_to_target are measured against.
        # The decomposition next to it says how much of it is base-rate
        # separation between tasks rather than within-task signal.
        log = {
            "val/auroc_pooled": auroc,
            "val/auroc_zero_shot": auroc,
            "val/mtp_loss": mtp_loss,
            "val/score_db": float("nan"),
        }
        log.update(
            decompose_val_auroc(preds_np, labels_bin, task_ids_np, self.task_names)
        )
        log.update(_mean_stats(stats_accum, stats_count, prefix="val_probe/"))

        eval_seconds = time.perf_counter() - eval_started
        if step == 0:
            log["timing/init_eval_seconds"] = eval_seconds
        else:
            self.eval_seconds_total += eval_seconds
            log["timing/val_eval_seconds"] = eval_seconds
            log["timing/val_eval_seconds_total"] = self.eval_seconds_total

        # Step 0 is the zero-gradient baseline: it must never set time_to_target
        # (the timer has not even started, so it would read ~0s and the metric
        # would be meaningless). Record the crossing as a flag instead — an init
        # that already clears the target is the finding, not a 0-second record.
        if step == 0:
            crossed = float(auroc >= self.TARGET_AUROC)
            log["val/target_crossed_at_init"] = crossed
            if crossed:
                print(
                    f"\n*** TARGET ALREADY CLEARED AT INIT (step 0, zero gradient "
                    f"steps): auroc={auroc:.4f} >= {self.TARGET_AUROC} — "
                    f"time_to_target is degenerate for this config ***"
                )

        if (
            step > 0
            and self._time_to_target is None
            and auroc >= self.TARGET_AUROC
            and self.timer is not None
        ):
            self._time_to_target = self.timer.elapsed()
            log["train/time_to_target"] = self._time_to_target
            print(
                f"\n*** TARGET REACHED: auroc={auroc:.4f} "
                f"time_to_target={self._time_to_target:.1f}s ***"
            )

        # Step 0 is a scientific control only: it cannot select a checkpoint.
        if step > 0 and self.save_ckpt_dir is not None and auroc > self.best_auroc:
            self.best_auroc = auroc
            ckpt_dir = Path(self.save_ckpt_dir).expanduser()
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(_canonical_state_dict(net), ckpt_dir / "best.pt")
            print(
                f"\n[ckpt] new best auroc={auroc:.4f} @ step={step} -> {ckpt_dir}/best.pt"
            )

        wandb.log(log, step=step)
        per_task = "  ".join(
            f"{name}={log[f'val/auroc/{name}']:.4f}"
            f"(p={log[f'val/base_rate/{name}']:.3f})"
            for name in sorted(
                k[len("val/auroc/") :] for k in log if k.startswith("val/auroc/")
            )
        )
        print(
            f"\nstep={step}  pooled={auroc:.4f}"
            f"  mean_per_task={log['val/auroc_mean_per_task']:.4f}"
            f"  between={log['val/auroc_between']:.4f}"
            f"  cross_weight={log['val/pooled_cross_weight']:.3f}"
            f"  pos_neg_tie_rate={log['val/pred_pos_neg_tie_rate']:.3f}"
            f"  val/mtp_loss={mtp_loss:.4f}"
            f"  eval_seconds={eval_seconds:.1f}"
            f"\n          {per_task}"
        )
        net.train()

    def finalize(self, step: int) -> None:
        """Log time_to_target=-1 if the target was never reached."""
        if self._time_to_target is None:
            wandb.log({"train/time_to_target": -1}, step=step)


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def all_gather_nd(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Gathers tensor arrays of different lengths in a list.
    The length dimension is 0. This supports any number of extra dimensions in the tensors.
    All the other dimensions should be equal between the tensors.
    Adapted from: https://stackoverflow.com/a/71433508

    Args:
        tensor (Tensor): Tensor to be broadcast from current process.

    Returns:
        list[Tensor]: List of tensors gathered from all processes.
    """
    world_size = dist.get_world_size()
    local_size = torch.tensor(tensor.size(), device=tensor.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)

    max_length = max(size[0] for size in all_sizes)

    length_diff = max_length.item() - local_size[0].item()
    if length_diff:
        pad_size = (length_diff, *tensor.size()[1:])
        padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat((tensor, padding))

    all_tensors_padded = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(all_tensors_padded, tensor)
    all_tensors = []
    for tensor_, size in zip(all_tensors_padded, all_sizes):
        all_tensors.append(tensor_[: size[0]])
    return all_tensors


def _stat_to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().item()
    return float(value)


def _accumulate_stats(accum: dict[str, float], stats: dict) -> None:
    for key, value in stats.items():
        accum[key] = accum.get(key, 0.0) + _stat_to_float(value)


def _mean_stats(accum: dict[str, float], count: int, prefix: str = "") -> dict:
    if count == 0:
        return {}
    return {f"{prefix}{key}": value / count for key, value in accum.items()}


def main(cfg: TrainConfig):
    cfg.validate()
    # Single config object (rt/config.py) is the only entry point — no CLI.
    # Unpacked into locals here so the body below reads with bare names.
    project = cfg.project
    eval_splits = cfg.eval_splits
    eval_freq = cfg.eval_freq
    eval_pow2 = cfg.eval_pow2
    max_eval_steps = cfg.max_eval_steps
    load_ckpt_path = cfg.load_ckpt_path
    save_ckpt_dir = cfg.save_ckpt_dir
    compile_ = cfg.compile_
    seed = cfg.seed
    train_tasks = cfg.train_tasks
    eval_tasks = cfg.eval_tasks
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    max_bfs_width = cfg.max_bfs_width
    lr = cfg.lr
    wd = cfg.wd
    lr_schedule = cfg.lr_schedule
    max_grad_norm = cfg.max_grad_norm
    max_steps = cfg.max_steps
    lr_total_steps = cfg.lr_total_steps
    optimizer = cfg.optimizer
    muon_lr = cfg.muon_lr
    muon_warmup_steps = cfg.muon_warmup_steps
    embedding_model = cfg.embedding_model
    d_text = cfg.d_text
    seq_len = cfg.seq_len
    num_blocks = cfg.num_blocks
    d_model = cfg.d_model
    num_heads = cfg.num_heads
    d_ff = cfg.d_ff
    simplicial_layers = cfg.simplicial_layers
    update_relations = cfg.update_relations
    qk_norm = cfg.qk_norm
    simplicial_layerscale_init = cfg.simplicial_layerscale_init
    sigreg_layer = cfg.sigreg_layer
    sigreg_lambda_row = cfg.sigreg_lambda_row
    sigreg_lambda_col = cfg.sigreg_lambda_col
    sigreg_lr = cfg.sigreg_lr
    sigreg_n_proj = cfg.sigreg_n_proj
    sigreg_proj_hidden_mult = cfg.sigreg_proj_hidden_mult
    sigreg_proj_out_div = cfg.sigreg_proj_out_div
    sigreg_proj_hidden_layers = cfg.sigreg_proj_hidden_layers
    sigreg_knots = cfg.sigreg_knots
    sigreg_t_max = cfg.sigreg_t_max
    sigreg_min_group = cfg.sigreg_min_group
    sigreg_col_global_pool = cfg.sigreg_col_global_pool
    val_tasks = cfg.val_tasks
    val_eval_freq = cfg.val_eval_freq
    time_budget_minutes = cfg.time_budget_minutes
    eval_at_step1 = cfg.eval_at_step1
    eval_at_step0 = cfg.eval_at_step0
    val_eval_dense_until = cfg.val_eval_dense_until
    val_eval_dense_freq = cfg.val_eval_dense_freq

    seed_everything(seed)

    ddp = "LOCAL_RANK" in os.environ
    device = "cuda"
    if ddp:
        os.environ["OMP_NUM_THREADS"] = f"{num_workers}"
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    if ddp:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if rank == 0:
        run_config = {**asdict(cfg), **_reproducibility_metadata(device)}
        run = wandb.init(project=project, config=run_config, tags=["nanort"])
        print(run.name)

    # NanoRT: initialise 45-minute training timer
    timer: TrainTimer | None = None
    if time_budget_minutes is not None:
        timer = TrainTimer(budget_minutes=time_budget_minutes)

    # NanoRT: initialise val AUROC callback (rank 0 only; not DDP-safe)
    val_callback: ValAUROCCallback | None = None
    if val_tasks and rank == 0 and not ddp:
        _freq = val_eval_freq if val_eval_freq is not None else max(1, max_steps // 100)
        val_callback = ValAUROCCallback(
            val_tasks=val_tasks,
            batch_size=batch_size,
            seq_len=seq_len,
            max_bfs_width=max_bfs_width,
            embedding_model=embedding_model,
            d_text=d_text,
            num_workers=num_workers,
            device="cuda",
            eval_every=_freq,
            timer=timer,
            eval_at_step1=eval_at_step1,
            dense_until=val_eval_dense_until,
            dense_freq=val_eval_dense_freq,
            save_ckpt_dir=save_ckpt_dir,
        )

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch._dynamo.config.cache_size_limit = 16
    torch._dynamo.config.compiled_autograd = compile_ if ddp else False
    torch._dynamo.config.optimize_ddp = True
    torch.set_num_threads(1)

    dataset = RelationalDataset(
        tasks=[
            (db_name, table_name, target_column, "train", columns_to_drop)
            for db_name, table_name, target_column, columns_to_drop in train_tasks
        ],
        batch_size=batch_size,
        seq_len=seq_len,
        rank=rank,
        world_size=world_size,
        max_bfs_width=max_bfs_width,
        embedding_model=embedding_model,
        d_text=d_text,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
        in_order=True,
    )

    eval_loaders = {}
    for db_name, table_name, target_column, columns_to_drop in eval_tasks:
        for split in eval_splits:
            eval_dataset = RelationalDataset(
                tasks=[(db_name, table_name, target_column, split, columns_to_drop)],
                batch_size=batch_size,
                seq_len=seq_len,
                rank=rank,
                world_size=world_size,
                max_bfs_width=max_bfs_width,
                embedding_model=embedding_model,
                d_text=d_text,
                seed=0,
            )
            eval_dataset.sampler.shuffle_py(0)
            eval_loaders[(db_name, table_name, split)] = DataLoader(
                eval_dataset,
                batch_size=None,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                pin_memory=True,
                in_order=True,
            )

    net = RelationalTransformer(
        num_blocks=num_blocks,
        d_model=d_model,
        d_text=d_text,
        num_heads=num_heads,
        d_ff=d_ff,
        simplicial_layers=simplicial_layers,
        update_relations=update_relations,
        qk_norm=qk_norm,
        simplicial_layerscale_init=simplicial_layerscale_init,
        sigreg_layer=sigreg_layer,
        sigreg_n_proj=sigreg_n_proj,
        sigreg_proj_hidden_mult=sigreg_proj_hidden_mult,
        sigreg_proj_out_div=sigreg_proj_out_div,
        sigreg_proj_hidden_layers=sigreg_proj_hidden_layers,
        sigreg_knots=sigreg_knots,
        sigreg_t_max=sigreg_t_max,
        sigreg_min_group=sigreg_min_group,
        sigreg_col_global_pool=sigreg_col_global_pool,
        sigreg_use_row=sigreg_lambda_row > 0.0,
        sigreg_use_col=sigreg_lambda_col > 0.0,
    )
    if load_ckpt_path is not None:
        load_ckpt_path = Path(load_ckpt_path).expanduser()
        _load_model_state(net, load_ckpt_path)

    net = net.to(device)
    net = net.to(torch.bfloat16)

    # SIGReg runs in fp32 (CF quadrature + BatchNorm are numerically sensitive):
    # cast the projector submodule back to fp32 after the global bf16 cast.
    sigreg_enabled = sigreg_layer is not None
    sigreg_active = sigreg_enabled and (sigreg_lambda_row + sigreg_lambda_col) > 0.0
    if sigreg_enabled:
        net.sigreg.float()

    # ── Optimiser setup ───────────────────────────────────────────────────────
    if optimizer == "muon":
        muon_params, adamw_params = build_muon_param_groups(net)
        if rank == 0:
            muon_count = sum(p.numel() for p in muon_params)
            adamw_count = sum(p.numel() for p in adamw_params)
            total = muon_count + adamw_count
            print(
                f"param_groups: muon={muon_count:_} ({100 * muon_count / total:.1f}%)"
                f"  adamw={adamw_count:_} ({100 * adamw_count / total:.1f}%)"
                f"  total={total:_}"
            )
        opt_muon = Muon(muon_params, lr=muon_lr)
        opt_adamw = optim.AdamW(
            adamw_params,
            lr=lr,
            weight_decay=wd,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True,
        )

        def _warmup_lr(step: int) -> float:
            return min(1.0, step / max(1, muon_warmup_steps))

        muon_sched = optim.lr_scheduler.LambdaLR(opt_muon, lr_lambda=_warmup_lr)

        if lr_schedule:
            adamw_sched = optim.lr_scheduler.OneCycleLR(
                opt_adamw,
                max_lr=lr,
                total_steps=lr_total_steps if lr_total_steps is not None else max_steps,
                pct_start=0.2,
                anneal_strategy="linear",
            )
    else:
        # SIGReg projector params are trained by a separate fp32 optimizer.
        main_params = [
            p for n, p in net.named_parameters() if not n.startswith("sigreg.")
        ]
        if rank == 0:
            param_count = sum(p.numel() for p in main_params)
            print(f"{param_count=:_}")
        opt = optim.AdamW(
            main_params,
            lr=lr,
            weight_decay=wd,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True,
        )
        if lr_schedule:
            lrs = optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=lr,
                total_steps=lr_total_steps if lr_total_steps is not None else max_steps,
                pct_start=0.2,
                anneal_strategy="linear",
            )

    # SIGReg projector — separate fp32 AdamW (kept out of the fused bf16 groups).
    # Built on the raw net before DDP/compile so net.sigreg.parameters() resolves.
    opt_sigreg = None
    if sigreg_active:
        opt_sigreg = optim.AdamW(
            net.sigreg.parameters(),
            lr=sigreg_lr if sigreg_lr is not None else lr,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=False,
        )
    if rank == 0:
        n_proj_params = (
            sum(p.numel() for p in net.sigreg.parameters()) if sigreg_enabled else 0
        )
        print(
            f"[sigreg] enabled={sigreg_enabled} active={sigreg_active} "
            f"layer={sigreg_layer} lambda_row={sigreg_lambda_row} "
            f"lambda_col={sigreg_lambda_col} projector_params={n_proj_params:_} "
            f"opt_sigreg={'yes' if opt_sigreg is not None else 'no'}",
            flush=True,
        )

    if ddp:
        net = DDP(net)

    net = torch.compile(net, dynamic=True, disable=not compile_)

    steps = 0
    if rank == 0:
        wandb.log({"epochs": 0}, step=steps)

    eval_loader_iters = {}
    for k, eval_loader in eval_loaders.items():
        eval_loader_iters[k] = iter(eval_loader)

    def evaluate(net):
        metrics = {"val": {}, "test": {}}
        net.eval()
        with torch.inference_mode():
            for (
                db_name,
                table_name,
                split,
            ), eval_loader_iter in eval_loader_iters.items():
                if table_name in [
                    "item-sales",
                    "user-ltv",
                    "item-ltv",
                    "post-votes",
                    "site-success",
                    "study-adverse",
                    "user-attendance",
                    "driver-position",
                    "ad-ctr",
                ]:
                    task_type = "reg"
                else:
                    task_type = "clf"

                preds = []
                labels = []
                losses = []
                eval_load_times = []
                stats_accum = {}
                stats_count = 0
                eval_loader = eval_loaders[(db_name, table_name, split)]
                pbar = tqdm(
                    total=(
                        min(max_eval_steps, len(eval_loader))
                        if max_eval_steps > -1
                        else len(eval_loader)
                    ),
                    desc=f"{db_name}/{table_name}/{split}",
                    disable=rank != 0,
                )

                batch_idx = 0
                while True:
                    tic = time.time()
                    try:
                        batch = next(eval_loader_iter)
                        batch_idx += 1
                    except StopIteration:
                        break
                    toc = time.time()
                    pbar.update(1)

                    eval_load_time = toc - tic
                    if rank == 0:
                        eval_load_times.append(eval_load_time)

                    true_batch_size = batch.pop("true_batch_size")
                    for k in batch:
                        batch[k] = batch[k].to(device, non_blocking=True)

                    batch["masks"][true_batch_size:, :] = False
                    batch["is_targets"][true_batch_size:, :] = False
                    batch["is_padding"][true_batch_size:, :] = True

                    loss, yhat_dict, eval_stats = net(batch)
                    if rank == 0:
                        _accumulate_stats(stats_accum, eval_stats)
                        stats_count += 1

                    if task_type == "clf":
                        yhat = yhat_dict["boolean"][batch["is_targets"]]
                        y = batch["boolean_values"][batch["is_targets"]].flatten()
                    elif task_type == "reg":
                        yhat = yhat_dict["number"][batch["is_targets"]]
                        y = batch["number_values"][batch["is_targets"]].flatten()
                    else:
                        raise ValueError(
                            f"Unsupported evaluation task type: {task_type}"
                        )

                    if yhat.size(0) != true_batch_size or y.size(0) != true_batch_size:
                        raise RuntimeError(
                            "Evaluation target count does not match true_batch_size: "
                            f"predictions={yhat.size(0)}, labels={y.size(0)}, "
                            f"expected={true_batch_size}."
                        )

                    pred = yhat.flatten()

                    losses.append(loss.item())
                    preds.append(pred)
                    labels.append(y)

                    if max_eval_steps > -1 and batch_idx >= max_eval_steps:
                        break

                eval_loader_iters[(db_name, table_name, split)] = iter(eval_loader)

                pbar.close()
                preds = torch.cat(preds, dim=0)
                labels = torch.cat(labels, dim=0)

                if ddp:
                    preds = all_gather_nd(preds)
                    labels = all_gather_nd(labels)
                else:
                    preds = [preds]
                    labels = [labels]

                if rank == 0:
                    loss = sum(losses) / len(losses)
                    k = f"loss/{db_name}/{table_name}/{split}"
                    avg_eval_load_time = sum(eval_load_times) / len(eval_load_times)
                    wandb.log(
                        {
                            k: loss,
                            f"avg_eval_load_time/{db_name}/{table_name}": avg_eval_load_time,
                            **_mean_stats(
                                stats_accum,
                                stats_count,
                                prefix=f"{split}/{db_name}/{table_name}/",
                            ),
                        },
                        step=steps,
                    )

                    preds = torch.cat(preds, dim=0).float().cpu().numpy()
                    labels = torch.cat(labels, dim=0).float().cpu().numpy()

                    if task_type == "reg":
                        metric_name = "r2"
                        metric = r2_score(labels, preds)
                    elif task_type == "clf":
                        metric_name = "auc"
                        labels = [int(x > 0) for x in labels]
                        metric = roc_auc_score(labels, preds)

                    k = f"{metric_name}/{db_name}/{table_name}/{split}"
                    wandb.log({k: metric}, step=steps)
                    print(f"\nstep={steps}, \t{k}: {metric}")
                    metrics[split][(db_name, table_name)] = metric

        return metrics

    def checkpoint(best=False, db_name="", table_name=""):
        if rank != 0 or save_ckpt_dir is None:
            return
        save_ckpt_dir_ = Path(save_ckpt_dir).expanduser()
        save_ckpt_dir_.mkdir(parents=True, exist_ok=True)
        if best:
            save_ckpt_path = f"{save_ckpt_dir_}/{db_name}_{table_name}_best.pt"
        else:
            save_ckpt_path = f"{save_ckpt_dir_}/{steps=}.pt"

        torch.save(_canonical_state_dict(net), save_ckpt_path)
        print(f"saved checkpoint to {save_ckpt_path}")

    def opt_zero_grad():
        if optimizer == "muon":
            opt_muon.zero_grad(set_to_none=True)
            opt_adamw.zero_grad(set_to_none=True)
        else:
            opt.zero_grad(set_to_none=True)
        if opt_sigreg is not None:
            opt_sigreg.zero_grad(set_to_none=True)

    def opt_step():
        if optimizer == "muon":
            opt_muon.step()
            opt_adamw.step()
            muon_sched.step()
            if lr_schedule:
                adamw_sched.step()
        else:
            opt.step()
            if lr_schedule:
                lrs.step()
        if opt_sigreg is not None:
            opt_sigreg.step()

    pbar = tqdm(total=max_steps, desc="steps", disable=rank != 0)

    best_val_metrics = {}
    best_test_metrics = {}
    done = False

    while steps < max_steps and not done:
        loader.dataset.sampler.shuffle_py(int(steps / len(loader)))
        loader_iter = iter(loader)
        while steps < max_steps and not done:
            if (eval_freq is not None and steps % eval_freq == 0) or (
                eval_pow2 and steps & (steps - 1) == 0
            ):
                metrics = evaluate(net)
                if save_ckpt_dir is not None:
                    for (db_name, table_name), metric in metrics["val"].items():
                        best_metric = best_val_metrics.get(
                            (db_name, table_name), -float("inf")
                        )
                        if metric > best_metric:
                            best_val_metrics[(db_name, table_name)] = metric
                            best_test_metrics[(db_name, table_name)] = metrics["test"][
                                (db_name, table_name)
                            ]
                            checkpoint(
                                best=True, db_name=db_name, table_name=table_name
                            )
                        else:
                            checkpoint()

            net.train()

            tic = time.time()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            batch.pop("true_batch_size")
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)
            toc = time.time()
            load_time = toc - tic
            if rank == 0:
                wandb.log({"load_time": load_time}, step=steps)

            # Stats (entropies, norms, uniques) force extra kernels and
            # GPU->CPU syncs — only compute them on steps that get logged.
            # `steps` is incremented after the forward, so mirror the log
            # condition below with steps + 1.
            need_stats = (steps + 1 <= 5) or ((steps + 1) % 10 == 0)

            if steps == 0 and compile_:
                # Warm up torch.compile for both need_stats variants during
                # step 1, which is excluded from the timer (timer.start()
                # only fires after the first step completes). Gradients are
                # discarded, no optimizer step: the training trajectory is
                # unchanged.
                #
                # The warm-up backward MUST include the SIGReg term. AOTAutograd
                # specializes the compiled backward on the outputs that receive
                # grad in the first backward it sees; warming with the bare
                # supervised loss bakes in a backward with no projector->backbone
                # path, so at real steps the SIGReg gradient silently never
                # reaches the backbone (loss logs fine, AUROC == no-SIGReg).
                for _warm_stats in (True, False):
                    warm_loss, _, warm_stats = net(batch, need_stats=_warm_stats)
                    if sigreg_active:
                        warm_loss = (
                            (1.0 - sigreg_lambda_row - sigreg_lambda_col) * warm_loss
                            + sigreg_lambda_row * warm_stats["sigreg_row"]
                            + sigreg_lambda_col * warm_stats["sigreg_col"]
                        )
                    warm_loss.backward()
                opt_zero_grad()

            # Zero-gradient baseline. Placed *after* the compile warm-up on
            # purpose: the warm-up backward has to be the first backward the
            # compiler sees (see above), and an eval-mode inference_mode forward
            # ahead of it would risk re-introducing the bug c0b1d52 fixed. No
            # optimizer step has run yet, so this measures the init exactly.
            if steps == 0 and val_callback is not None and eval_at_step0:
                val_callback.step(0, net)

            loss, _, train_stats = net(batch, need_stats=need_stats)
            # SIGReg: L = (1-λc-λr)·L_sup + λc·R_col + λr·R_row (paper §2.4).
            # A single backward updates both the backbone and the projector.
            if sigreg_active:
                row_loss = train_stats.get(
                    "sigreg_row", loss.new_zeros((), dtype=torch.float32)
                )
                col_loss = train_stats.get(
                    "sigreg_col", loss.new_zeros((), dtype=torch.float32)
                )
                sup_weighted = (1.0 - sigreg_lambda_row - sigreg_lambda_col) * loss
                row_weighted = sigreg_lambda_row * row_loss
                col_weighted = sigreg_lambda_col * col_loss
                total_loss = sup_weighted + row_weighted + col_weighted
            else:
                total_loss = loss
            opt_zero_grad()
            total_loss.backward()

            grad_norm = get_total_norm(
                [p.grad for p in net.parameters() if p.grad is not None]
            )
            clip_grads_with_norm_(
                net.parameters(), max_norm=max_grad_norm, total_norm=grad_norm
            )

            opt_step()
            steps += 1

            if ddp:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            if rank == 0:
                log = {
                    "loss": loss,
                    "epochs": steps / len(loader),
                    "grad_norm": grad_norm,
                }
                if optimizer == "muon":
                    log["lr_muon"] = opt_muon.param_groups[0]["lr"]
                    log["lr_adamw"] = opt_adamw.param_groups[0]["lr"]
                else:
                    log["lr"] = opt.param_groups[0]["lr"]
                # Log simplicial attention stats early, then periodically.
                if train_stats and (steps <= 5 or steps % 10 == 0):
                    flat_train_stats = {
                        f"train/{k}": _stat_to_float(v) for k, v in train_stats.items()
                    }
                    log.update(flat_train_stats)
                    # simplicial/* keys are absent when the simplicial path is
                    # disabled (simplicial_layers=none) — guard so that control
                    # run doesn't KeyError at the first log step.
                    if (
                        steps <= 5 or steps % 50 == 0
                    ) and "train/simplicial/edge_count" in flat_train_stats:
                        print(
                            f"\nstep={steps}  "
                            f"train/simplicial/edge_count="
                            f"{flat_train_stats['train/simplicial/edge_count']:.0f}  "
                            f"parent_match_rate="
                            f"{flat_train_stats['train/simplicial/parent_match_rate']:.3f}  "
                            f"active_relation_types="
                            f"{flat_train_stats['train/simplicial/active_relation_types']:.0f}  "
                            f"nbr_delta_frac="
                            f"{flat_train_stats['train/simplicial/nbr_delta_frac']:.3f}  "
                            f"rel_delta_frac="
                            f"{flat_train_stats['train/simplicial/rel_delta_frac']:.3f}"
                        )
                wandb.log(log, step=steps)

            # NanoRT: timer — start on first step, log elapsed, enforce budget
            if timer is not None:
                timer.start()
                if rank == 0:
                    wandb.log({"train/elapsed_seconds": timer.elapsed()}, step=steps)
                if timer.over_budget():
                    checkpoint()
                    done = True

            # NanoRT: zero-shot AUROC evaluation
            if val_callback is not None and not done:
                val_callback.step(steps, net)

            pbar.update(1)

    if val_callback is not None:
        val_callback.finalize(steps)

    if rank == 0:
        print("\n" + "=" * 80)
        print("Best test metrics:")
        print("=" * 80)
        for (db_name, table_name), metric in best_test_metrics.items():
            print(f"{db_name}/{table_name}/test: {metric:.4f}")
        print("=" * 80 + "\n")

    if ddp:
        dist.destroy_process_group()
    if rank == 0:
        wandb.finish()
