"""CPU checks for optimizer updates and the benchmark timing boundary."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from rt import main as training
from rt.config import TrainConfig


def run_small_training(
    monkeypatch, module, *, optimizer, compile_, budget_stop=False, lr_schedule=False
):
    events, models = [], []

    class Dataset:
        def __init__(self, **kwargs):
            self.sampler = SimpleNamespace(shuffle_py=lambda epoch: None)

        def __len__(self):
            return 2

    class Loader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset

        def __len__(self):
            return 2

        def __iter__(self):
            return iter(
                [
                    {"x": torch.tensor([[1.0], [2.0]]), "true_batch_size": 2}
                    for _ in range(2)
                ]
            )

    class Model(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [torch.nn.ModuleDict({"ffn": torch.nn.Linear(1, 1, bias=False)})]
            )
            self.offset = torch.nn.Parameter(torch.randn(1))
            self.sigreg = torch.nn.Linear(1, 1, bias=False)
            self.initial = {
                key: value.detach().clone() for key, value in self.state_dict().items()
            }
            models.append(self)

        def to(self, *args, **kwargs):
            return self  # The driver still requests CUDA/bfloat16 in production.

        def forward(self, batch, **kwargs):
            prediction = self.blocks[0]["ffn"](batch["x"]) + self.offset
            return (
                prediction.square().mean(),
                {},
                {
                    "sigreg_row": prediction.new_zeros(()),
                    "sigreg_col": self.sigreg(prediction).square().mean(),
                },
            )

    class Timer:
        def __init__(self, **kwargs):
            self.started = False
            self.updates = 0

        def start(self):
            self.started = True
            self.updates += 1
            events.append(("clock", self.updates))

        def elapsed(self):
            return 0.0

        def over_budget(self):
            return budget_stop and self.updates == 2

    class Validation:
        def __init__(self, timer, **kwargs):
            self.timer = timer

        def step(self, step, net):
            if step == 0:
                assert not self.timer.started
            else:
                assert self.timer.started
            events.append(("evaluation", step))

        def finalize(self, step):
            events.append(("finalize", step))

    tensor_to, adamw = torch.Tensor.to, torch.optim.AdamW

    def cpu_to(self, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0] == "cuda":
            args = ("cpu", *args[1:])
        return tensor_to(self, *args, **kwargs)

    def cpu_adamw(*args, **kwargs):
        kwargs.pop("fused", None)
        return adamw(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: None)
    monkeypatch.setattr(module.optim, "AdamW", cpu_adamw)
    monkeypatch.setattr(module.torch, "compile", lambda model, **kwargs: model)
    monkeypatch.setattr(module, "_reproducibility_metadata", lambda device: {})
    monkeypatch.setattr(module, "RelationalDataset", Dataset)
    monkeypatch.setattr(module, "DataLoader", Loader)
    monkeypatch.setattr(module, "RelationalTransformer", Model)
    monkeypatch.setattr(module, "TrainTimer", Timer)
    monkeypatch.setattr(module, "ValAUROCCallback", Validation)
    monkeypatch.setattr(
        module, "wandb", Mock(init=Mock(return_value=SimpleNamespace(name="test")))
    )
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    cfg = TrainConfig(
        optimizer=optimizer,
        compile_=compile_,
        max_steps=8 if lr_schedule else 3,
        lr_schedule=lr_schedule,
    )
    cfg.train_tasks = [("train", "table", "target", [])]
    cfg.val_tasks = [("holdout", "table", "target", "Val", [])]
    module.main(cfg)
    state = models[0].state_dict()
    assert all(
        not torch.equal(state[key], value) for key, value in models[0].initial.items()
    )
    return state, events


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
@pytest.mark.parametrize("compile_", [False, True])
@pytest.mark.parametrize("lr_schedule", [False, True])
def test_driver_updates_backbone_and_projector(
    monkeypatch, optimizer, compile_, lr_schedule
):
    state, events = run_small_training(
        monkeypatch,
        training,
        optimizer=optimizer,
        compile_=compile_,
        lr_schedule=lr_schedule,
    )
    assert all(torch.isfinite(value).all() for value in state.values())
    expected = [("evaluation", 0)]
    steps = 8 if lr_schedule else 3
    for step in range(1, steps + 1):
        expected.extend([("clock", step), ("evaluation", step)])
    expected.append(("finalize", steps))
    assert events == expected


def test_budget_exhaustion_skips_later_validation(monkeypatch):
    _, events = run_small_training(
        monkeypatch,
        training,
        optimizer="muon",
        compile_=False,
        budget_stop=True,
    )
    assert events[-2:] == [("clock", 2), ("finalize", 2)]


def test_step_zero_cannot_select_checkpoint_or_record(monkeypatch, tmp_path):
    batch = {
        "true_batch_size": 2,
        "masks": torch.ones(3, 1, dtype=torch.bool),
        "is_targets": torch.ones(3, 1, dtype=torch.bool),
        "is_padding": torch.zeros(3, 1, dtype=torch.bool),
        "boolean_values": torch.tensor([[[1.0]], [[0.0]], [[0.0]]]).reshape(3, 1, 1),
        "col_name_idxs": torch.tensor([[10], [20], [99]]),
    }
    dataset = SimpleNamespace(sampler=SimpleNamespace(shuffle_py=lambda epoch: None))
    monkeypatch.setattr(training, "RelationalDataset", lambda **kwargs: dataset)

    class ValidationLoader:
        def __iter__(self):
            yield {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }

    monkeypatch.setattr(
        training, "DataLoader", lambda *args, **kwargs: ValidationLoader()
    )
    monkeypatch.setattr(
        training,
        "get_column_index",
        lambda target, table, db: {"a": 10, "b": 20}[table],
    )
    monkeypatch.setattr(training, "wandb", Mock())

    class Model(torch.nn.Module):
        def forward(self, batch, **kwargs):
            return (
                torch.tensor(0.0),
                {"boolean": torch.tensor([[[3.0]], [[-1.0]], [[99.0]]])},
                {},
            )

    callback = training.ValAUROCCallback(
        val_tasks=[("db", table, "target", "Val", []) for table in ("a", "b")],
        batch_size=3,
        seq_len=1,
        max_bfs_width=1,
        embedding_model="test",
        d_text=1,
        num_workers=0,
        device="cpu",
        eval_every=100,
        timer=SimpleNamespace(elapsed=lambda: 12.5),
        save_ckpt_dir=str(tmp_path),
    )
    net = Model()
    callback.step(0, net)
    assert callback._time_to_target is None
    assert not (tmp_path / "best.pt").exists()
    callback.step(100, net)
    assert callback._time_to_target == 12.5
    assert callback.best_auroc == 1.0
    assert (tmp_path / "best.pt").is_file()
    assert net.training
