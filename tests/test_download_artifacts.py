"""Tests for the relation-aware Hugging Face artifact installer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data.download import (
    DATABASES,
    REQUIRED_FILES,
    artifact_prefix,
    download_patterns,
    embedding_filename,
    install_links,
    validate_bundle,
)


class DownloadArtifactsTest(unittest.TestCase):
    def _write_database(self, path: Path) -> None:
        path.mkdir(parents=True)
        for name in REQUIRED_FILES:
            (path / name).write_bytes(b"x")
        (path / "relation_text_idxs.json").write_text(json.dumps([0, 2]))
        # Three bfloat16 rows of width four.
        (path / embedding_filename("test-embedding")).write_bytes(b"0" * 24)

    def _write_bundle(self, root: Path, holdout: str = "rel-avito") -> None:
        prefix = root / artifact_prefix(holdout)
        for db in DATABASES:
            if db == holdout:
                self._write_database(prefix / "val" / db)
            else:
                self._write_database(prefix / "train" / db)

    def test_patterns_download_only_runtime_files(self):
        patterns = download_patterns("rel-avito", "test-embedding")
        self.assertEqual(len(patterns), len(DATABASES) * (len(REQUIRED_FILES) + 1))
        self.assertTrue(
            all(pattern.startswith("ra-holdout-rel-avito/") for pattern in patterns)
        )
        self.assertFalse(any("text.json" in pattern for pattern in patterns))

    def test_validate_and_install_relation_aware_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_root = root / "artifacts"
            install_root = root / "pre"
            self._write_bundle(artifact_root)

            paths = validate_bundle(
                artifact_root,
                holdout="rel-avito",
                embedding_model="test-embedding",
                d_text=4,
            )
            install_links(paths, install_root)

            self.assertEqual(set(paths), set(DATABASES))
            self.assertTrue((install_root / "rel-avito").is_symlink())
            self.assertIn("/val/", str((install_root / "rel-avito").resolve()))
            self.assertIn("/train/", str((install_root / "rel-amazon").resolve()))

    def test_validate_rejects_holdout_in_training_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory)
            self._write_bundle(artifact_root)
            extra = artifact_root / artifact_prefix("rel-avito") / "train" / "rel-avito"
            self._write_database(extra)

            with self.assertRaisesRegex(ValueError, "holdout isolation"):
                validate_bundle(
                    artifact_root,
                    holdout="rel-avito",
                    embedding_model="test-embedding",
                    d_text=4,
                )


if __name__ == "__main__":
    unittest.main()
