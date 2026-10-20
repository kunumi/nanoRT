"""Check the public launcher without downloading data or requiring a GPU."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_launcher_builds_installs_and_forwards_configuration(tmp_path):
    fake_pixi = tmp_path / "pixi"
    calls_path = tmp_path / "calls.jsonl"
    fake_pixi.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['LAUNCHER_TEST_LOG'], 'a') as file:\n"
        "    file.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), "
        "'seed': os.environ['NANORT_SEED'], 'holdout': os.environ['NANORT_HOLDOUT_DB'], "
        "'logging': os.environ['WANDB_MODE']}) + '\\n')\n"
    )
    fake_pixi.chmod(0o755)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("NANORT_", "HF_", "WANDB_"))
    }
    environment.update(
        PATH=f"{tmp_path}:{environment['PATH']}",
        LAUNCHER_TEST_LOG=str(calls_path),
        NANORT_SEED="7",
        NANORT_HOLDOUT_DB="rel-amazon",
    )
    subprocess.run(
        ["bash", str(ROOT / "run.sh")], cwd=tmp_path, env=environment, check=True
    )
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert [call["args"][:2] for call in calls] == [
        ["install", "--locked"],
        ["run", "maturin"],
        ["run", "python"],
        ["run", "python"],
    ]
    assert calls[1]["cwd"] == str(ROOT / "rustler")
    assert calls[2]["args"] == [
        "run",
        "python",
        "data/download.py",
        "--repo-id",
        "GabrielVeloso/nanort-artifacts",
        "--revision",
        "8e6289edd0aa1d61b3b4c2ec1e1beed48224cd2c",
    ]
    assert calls[3]["args"] == ["run", "python", "-u", "scripts/nanort_train.py"]
    assert all(
        call["seed"] == "7" and call["holdout"] == "rel-amazon" for call in calls
    )
    assert all(call["logging"] == "offline" for call in calls)


def test_launcher_rejects_positional_configuration():
    result = subprocess.run(
        ["bash", str(ROOT / "run.sh"), "--seed=7"], capture_output=True
    )
    assert result.returncode == 2
    assert b"environment variables only" in result.stderr
