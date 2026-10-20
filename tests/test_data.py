"""The semantic FK vectors must retain their frozen bfloat16 representation."""

import json

import ml_dtypes
import numpy as np

from rt import data


def test_frozen_relation_indices_select_exact_embedding_rows(monkeypatch, tmp_path):
    database = tmp_path / "fixture"
    database.mkdir()
    (database / "relation_text_idxs.json").write_text(json.dumps([0, 2]))
    values = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=ml_dtypes.bfloat16)
    values.tofile(database / "text_emb_test.bin")
    monkeypatch.setenv("NANORT_PRE_ROOT", str(tmp_path))
    try:
        selected = data._load_relation_embeddings("fixture", "test", 2)
        assert selected.dtype == values.dtype
        np.testing.assert_array_equal(
            selected.view(np.uint16), values[[0, 2]].view(np.uint16)
        )
        assert not data._load_text_embeddings("fixture", "test", 2).flags.writeable
    finally:
        data._load_relation_text_idxs.cache_clear()
        data._load_text_embeddings.cache_clear()
