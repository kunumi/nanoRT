import os
import platform
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import optim
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
    """Return the raw module behind ``torch.compile``."""
    while True:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


def _canonical_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Save weights without the compiler wrapper prefix."""
    return _unwrap_model(model).state_dict()


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


class ValAUROCCallback:
    """Evaluate holdout AUROC, track target time, and save the best weights."""

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
        # Map target-column IDs to task names for AUROC diagnostics.
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
        # Step 1 is optional; record eligibility starts at step 100 by default.
        self.eval_at_step1 = eval_at_step1
        # Optionally use a denser early evaluation cadence.
        self.dense_until = dense_until
        self.dense_freq = max(1, dense_freq)
        # Save the best eligible post-update checkpoint.
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

                # Probe diagnostics on the first batch; score the complete holdout.
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

        # Keep the historical pooled-metric alias for archived runs.
        log = {
            "val/auroc_pooled": auroc,
            "val/auroc_zero_shot": auroc,
            "val/mtp_loss": mtp_loss,
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

        # Step 0 is diagnostic and cannot establish a record.
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


def _training_loss(loss, stats, cfg):
    return (
        (1.0 - cfg.sigreg_lambda_row - cfg.sigreg_lambda_col) * loss
        + cfg.sigreg_lambda_row * stats["sigreg_row"]
        + cfg.sigreg_lambda_col * stats["sigreg_col"]
    )


def main(cfg: TrainConfig):
    cfg.validate()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This training entry point supports one GPU only.")
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU; the benchmark uses one H200.")
    device = "cuda"
    seed_everything(cfg.seed)
    run = wandb.init(
        project=cfg.project,
        config={**asdict(cfg), **_reproducibility_metadata(device)},
        tags=["nanort"],
    )
    print(run.name)
    timer = (
        TrainTimer(budget_minutes=cfg.time_budget_minutes)
        if cfg.time_budget_minutes
        else None
    )
    validation = ValAUROCCallback(
        val_tasks=cfg.val_tasks,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        max_bfs_width=cfg.max_bfs_width,
        embedding_model=cfg.embedding_model,
        d_text=cfg.d_text,
        num_workers=cfg.num_workers,
        device=device,
        eval_every=cfg.val_eval_freq,
        timer=timer,
        eval_at_step1=cfg.eval_at_step1,
        dense_until=cfg.val_eval_dense_until,
        dense_freq=cfg.val_eval_dense_freq,
        save_ckpt_dir=cfg.save_ckpt_dir,
    )

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch._dynamo.config.cache_size_limit = 16
    torch.set_num_threads(1)
    dataset = RelationalDataset(
        tasks=[
            (db, table, target, "train", drop)
            for db, table, target, drop in cfg.train_tasks
        ],
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        max_bfs_width=cfg.max_bfs_width,
        embedding_model=cfg.embedding_model,
        d_text=cfg.d_text,
        seed=cfg.seed,
    )
    if len(dataset) == 0:
        raise ValueError("The training artifact contains no eligible examples.")
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=cfg.num_workers,
        persistent_workers=False,
        pin_memory=True,
        in_order=True,
    )
    net = (
        RelationalTransformer(
            num_blocks=cfg.num_blocks,
            d_model=cfg.d_model,
            d_text=cfg.d_text,
            num_heads=cfg.num_heads,
            d_ff=cfg.d_ff,
            simplicial_layers=cfg.simplicial_layers,
            update_relations=cfg.update_relations,
            qk_norm=cfg.qk_norm,
            simplicial_layerscale_init=cfg.simplicial_layerscale_init,
            sigreg_layer=cfg.sigreg_layer,
            sigreg_n_proj=cfg.sigreg_n_proj,
            sigreg_proj_hidden_mult=cfg.sigreg_proj_hidden_mult,
            sigreg_proj_out_div=cfg.sigreg_proj_out_div,
            sigreg_proj_hidden_layers=cfg.sigreg_proj_hidden_layers,
            sigreg_knots=cfg.sigreg_knots,
            sigreg_t_max=cfg.sigreg_t_max,
            sigreg_min_group=cfg.sigreg_min_group,
            sigreg_col_global_pool=cfg.sigreg_col_global_pool,
            sigreg_use_row=cfg.sigreg_lambda_row > 0,
            sigreg_use_col=cfg.sigreg_lambda_col > 0,
        )
        .to(device)
        .to(torch.bfloat16)
    )
    # The SIGReg quadrature and projector remain fp32.
    if net.sigreg is not None:
        net.sigreg.float()
    sigreg_active = cfg.sigreg_lambda_row + cfg.sigreg_lambda_col > 0

    optimizers = {}
    schedulers = []
    if cfg.optimizer == "muon":
        muon_params, adamw_params = build_muon_param_groups(net)
        optimizers["lr_muon"] = Muon(muon_params, lr=cfg.muon_lr)
        schedulers.append(
            optim.lr_scheduler.LambdaLR(
                optimizers["lr_muon"],
                lr_lambda=lambda step: min(1.0, step / max(1, cfg.muon_warmup_steps)),
            )
        )
        adamw_key = "lr_adamw"
    else:
        adamw_params = [
            p for n, p in net.named_parameters() if not n.startswith("sigreg.")
        ]
        adamw_key = "lr"
    optimizers[adamw_key] = optim.AdamW(
        adamw_params,
        lr=cfg.lr,
        weight_decay=cfg.wd,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
    )
    if cfg.lr_schedule:
        schedulers.append(
            optim.lr_scheduler.OneCycleLR(
                optimizers[adamw_key],
                max_lr=cfg.lr,
                total_steps=cfg.lr_total_steps or cfg.max_steps,
                pct_start=0.2,
                anneal_strategy="linear",
            )
        )
    if sigreg_active:
        optimizers["lr_sigreg"] = optim.AdamW(
            net.sigreg.parameters(),
            lr=cfg.sigreg_lr or cfg.lr,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=False,
        )

    def zero_grad():
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)

    net = torch.compile(net, dynamic=True, disable=not cfg.compile_)
    step = 0
    done = False
    wandb.log({"epochs": 0}, step=0)
    with tqdm(total=cfg.max_steps, desc="steps") as progress:
        while step < cfg.max_steps and not done:
            dataset.sampler.shuffle_py(step // len(loader))
            loader_iter = iter(loader)
            while step < cfg.max_steps and not done:
                net.train()
                load_started = time.time()
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break
                batch.pop("true_batch_size")
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items()
                }
                wandb.log({"load_time": time.time() - load_started}, step=step)
                need_stats = step + 1 <= 5 or (step + 1) % 10 == 0
                if step == 0 and cfg.compile_:
                    # Compile both paths with the complete loss. Omitting SIGReg
                    # here can specialize a backward without its backbone gradient.
                    for warm_stats in (True, False):
                        loss, _, stats = net(batch, need_stats=warm_stats)
                        (
                            _training_loss(loss, stats, cfg) if sigreg_active else loss
                        ).backward()
                    zero_grad()
                if step == 0 and cfg.eval_at_step0:
                    validation.step(0, net)

                loss, _, stats = net(batch, need_stats=need_stats)
                total_loss = _training_loss(loss, stats, cfg) if sigreg_active else loss
                zero_grad()
                total_loss.backward()
                grad_norm = get_total_norm(
                    [p.grad for p in net.parameters() if p.grad is not None]
                )
                clip_grads_with_norm_(
                    net.parameters(), max_norm=cfg.max_grad_norm, total_norm=grad_norm
                )
                for optimizer in optimizers.values():
                    optimizer.step()
                for scheduler in schedulers:
                    scheduler.step()
                step += 1
                log = {
                    "loss": loss,
                    "epochs": step / len(loader),
                    "grad_norm": grad_norm,
                }
                log.update(
                    {key: opt.param_groups[0]["lr"] for key, opt in optimizers.items()}
                )
                if need_stats:
                    log.update(
                        {
                            f"train/{key}": _stat_to_float(value)
                            for key, value in stats.items()
                        }
                    )
                wandb.log(log, step=step)

                # Start the historical timer after the first update and its logging.
                if timer is not None:
                    timer.start()
                    wandb.log({"train/elapsed_seconds": timer.elapsed()}, step=step)
                    if timer.over_budget():
                        if cfg.save_ckpt_dir:
                            directory = Path(cfg.save_ckpt_dir).expanduser()
                            directory.mkdir(parents=True, exist_ok=True)
                            torch.save(
                                _canonical_state_dict(net),
                                directory / f"steps={step}.pt",
                            )
                        done = True
                if not done:
                    validation.step(step, net)
                progress.update(1)
    validation.finalize(step)
    wandb.finish()
