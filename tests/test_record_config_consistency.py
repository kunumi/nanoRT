"""Keep every public record launcher aligned with ``TrainConfig`` defaults."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from rt.config import TrainConfig

ROOT = Path(__file__).resolve().parents[1]


def _run_sh_defaults() -> dict[str, str]:
    text = (ROOT / "run.sh").read_text()
    return dict(
        re.findall(
            r'^export (NANORT_[A-Z0-9_]+)="\$\{\1:-([^}]*)\}"$',
            text,
            flags=re.MULTILINE,
        )
    )


class RecordConfigConsistencyTest(unittest.TestCase):
    def test_shell_launcher_matches_record_defaults(self):
        cfg = TrainConfig()
        expected = {
            "NANORT_BATCH_SIZE": str(cfg.batch_size),
            "NANORT_OPTIMIZER": cfg.optimizer,
            "NANORT_LR": str(cfg.lr),
            "NANORT_MUON_LR": str(cfg.muon_lr),
            "NANORT_MUON_WARMUP_STEPS": str(cfg.muon_warmup_steps),
            "NANORT_SIMPLICIAL_LAYERS": "all",
            "NANORT_UPDATE_RELATIONS": "0",
            "NANORT_QK_NORM": "1",
            "NANORT_SIMPLICIAL_LAYERSCALE_INIT": str(cfg.simplicial_layerscale_init),
            "NANORT_SIGREG_LAYER": str(cfg.sigreg_layer),
            "NANORT_SIGREG_LAMBDA_ROW": str(cfg.sigreg_lambda_row),
            "NANORT_SIGREG_LAMBDA_COL": str(cfg.sigreg_lambda_col),
            "NANORT_SIGREG_N_PROJ": str(cfg.sigreg_n_proj),
            "NANORT_SIGREG_MIN_GROUP": str(cfg.sigreg_min_group),
            "NANORT_SIGREG_COL_POOL": "1",
            "NANORT_EVAL_AT_STEP0": "1",
            "NANORT_EVAL_AT_STEP1": "0",
            "NANORT_VAL_EVAL_FREQ": str(cfg.val_eval_freq),
            "NANORT_VAL_EVAL_DENSE_UNTIL": str(cfg.val_eval_dense_until),
            "NANORT_TIME_BUDGET_MINUTES": str(int(cfg.time_budget_minutes)),
        }
        defaults = _run_sh_defaults()
        self.assertEqual({key: defaults.get(key) for key in expected}, expected)

    def test_release_contains_no_experiment_launchers(self):
        forbidden = [path.name for path in ROOT.glob("plx_*")]
        for name in ("paper", "experiments", ".polyaxonignore"):
            if (ROOT / name).exists():
                forbidden.append(name)
        self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()
