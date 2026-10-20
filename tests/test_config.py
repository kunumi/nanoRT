"""CPU-only tests for the centralized nanoRT configuration."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from rt.config import TrainConfig


class TrainConfigTest(unittest.TestCase):
    def test_defaults_are_valid(self):
        self.assertIsInstance(TrainConfig().validate(), TrainConfig)

    def test_empty_environment_reproduces_record_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = TrainConfig.from_env()
        self.assertEqual(cfg.optimizer, "muon")
        self.assertIsNone(cfg.simplicial_layers)
        self.assertFalse(cfg.update_relations)
        self.assertEqual(cfg.sigreg_layer, 0)
        self.assertEqual(cfg.sigreg_lambda_col, 0.001)
        self.assertEqual(cfg.muon_warmup_steps, 3000)
        self.assertTrue(cfg.eval_at_step0)
        self.assertFalse(cfg.eval_at_step1)
        self.assertEqual(cfg.val_eval_freq, 100)

    def test_environment_parses_record_controls(self):
        env = {
            "NANORT_SIMPLICIAL_LAYERS": "all",
            "NANORT_UPDATE_RELATIONS": "0",
            "NANORT_MUON_WARMUP_STEPS": "3000",
            "NANORT_SIGREG_LAYER": "0",
            "NANORT_SIGREG_LAMBDA_COL": "0.001",
            "NANORT_EVAL_AT_STEP0": "1",
            "NANORT_EVAL_AT_STEP1": "0",
            "NANORT_VAL_EVAL_FREQ": "100",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = TrainConfig.from_env()
        self.assertIsNone(cfg.simplicial_layers)
        self.assertFalse(cfg.update_relations)
        self.assertEqual(cfg.muon_warmup_steps, 3000)
        self.assertEqual(cfg.sigreg_layer, 0)
        self.assertEqual(cfg.sigreg_lambda_col, 0.001)
        self.assertTrue(cfg.eval_at_step0)
        self.assertFalse(cfg.eval_at_step1)
        self.assertEqual(cfg.val_eval_freq, 100)

    def test_positive_sigreg_lambda_requires_layer(self):
        with self.assertRaisesRegex(ValueError, "requires sigreg_layer"):
            TrainConfig(sigreg_layer=None, sigreg_lambda_col=0.001).validate()

    def test_attention_dimensions_must_divide(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            TrainConfig(d_model=250, num_heads=8).validate()

    def test_simplicial_layers_must_be_in_range(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            TrainConfig(num_blocks=12, simplicial_layers=[12]).validate()

    def test_scheduler_cannot_end_before_training(self):
        with self.assertRaisesRegex(ValueError, "scheduler overrun"):
            TrainConfig(max_steps=100, lr_total_steps=99).validate()

    def test_invalid_evaluation_limits_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "val_eval_freq"):
            TrainConfig(val_eval_freq=0).validate()


if __name__ == "__main__":
    unittest.main()
