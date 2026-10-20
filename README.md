# modded-nanoRT

Code for **Speedrunning Relational Foundation Model Pretraining**.
The implementation combines a Relational Transformer with typed trilinear
attention over foreign-key edges, Muon and AdamW optimization, and
column-grouped SIGReg regularization.

## Quick start

Training requires Linux x86-64, one NVIDIA H200, a CUDA 12.4-compatible driver,
and [pixi](https://pixi.sh/).

```bash
git clone https://github.com/kunumi/nanoRT.git
cd nanoRT
./run.sh
```

The launcher installs the locked Python 3.12 / PyTorch 2.6 environment, builds
the Rust sampler, downloads the preprocessed data, checks holdout isolation,
and starts training. Initial setup downloads tens of gigabytes and compiles
the sampler. Logs are stored locally with W&B in offline mode by default.
Set `WANDB_MODE=online` to send logs to your W&B account.

The default dataset is
[GabrielVeloso/nanort-artifacts](https://huggingface.co/datasets/GabrielVeloso/nanort-artifacts),
pinned to commit `8e6289edd0aa1d61b3b4c2ec1e1beed48224cd2c`.
The installer records the resolved revision in the training metadata.

## Configuration

[`rt/config.py`](rt/config.py) defines the training configuration. Set
`NANORT_*` environment variables to override defaults:

```bash
NANORT_SEED=1 ./run.sh
NANORT_HOLDOUT_DB=rel-amazon ./run.sh
NANORT_TIME_BUDGET_MINUTES=10 ./run.sh
```

| Setting | Default |
|---|---|
| Training / validation | five RelBench databases / `rel-avito` |
| Optimizer | Muon for block matrices; AdamW for other parameters |
| Batch size / context length | `256` / `1024` |
| Model | 12 blocks, width 256, 8 heads |
| Typed trilinear attention | all blocks; static relation states; QK normalization |
| SIGReg | column groups after block 0, weight `1e-3` |
| Evaluation | untimed step 0, then every 100 updates |
| Target / budget | pooled AUROC `0.81621` / 45 minutes |

The implementation retains `simplicial` in class and configuration names for
compatibility. This names the typed trilinear operator used by the model.
See the [benchmark specification](docs/BENCHMARK.md) for the metric, timing
boundary, and holdout protocol.

## Data

To install the artifacts separately:

```bash
pixi install --locked
pixi run python data/download.py \
  --repo-id GabrielVeloso/nanort-artifacts \
  --revision 8e6289edd0aa1d61b3b4c2ec1e1beed48224cd2c \
  --holdout rel-avito
```

`NANORT_ARTIFACT_ROOT` controls the download cache; `NANORT_PRE_ROOT` controls
the installed data directory. `run.sh` defaults to
`~/.cache/modded-nanort/artifacts` and `~/.cache/modded-nanort/pre`.
Use `NANORT_HF_REPO` and `NANORT_HF_REVISION` to select another compatible bundle.
See the [data guide](data/README.md) and [artifact specification](docs/ARTIFACTS.md).

## Validation

On the supported Linux environment:

```bash
pixi install --locked
pixi run python -m pytest tests
pixi run python -m compileall -q data rt scripts tests
pixi run cargo test --locked --manifest-path rustler/Cargo.toml
bash -n run.sh docker_build.sh
```

The Python tests cover configuration, artifact installation, holdout isolation,
AUROC decomposition, attention, SIGReg, and model forward/backward gradients. Reproducing the full training run
and timing requires the H200 environment and the frozen data.

## Docker environment

Build the Linux training environment with:

```bash
./docker_build.sh
```

Mount the repository and cache directories when running the image:

```bash
docker run --rm --gpus all --ipc=host \
  -v "$PWD:/workspace" \
  -v nanort-cache:/root/.cache/modded-nanort \
  -w /workspace modded-nanort:cu124-torch2.6 bash run.sh
```

## Repository

```text
rt/          model, optimization, metrics, and training
rustler/     Rust preprocessing and context sampler
data/        artifact installer and data documentation
scripts/     training entry point
tests/       unit and protocol tests
docs/        data format and benchmark specifications
```

The backbone derives from the
[Relational Transformer](https://github.com/snap-stanford/relational-transformer).
The speedrun format follows
[modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) and
[modded-nanoTabPFN](https://github.com/borawhocodess/modded-nanotabpfn).
