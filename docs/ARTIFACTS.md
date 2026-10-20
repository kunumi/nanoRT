# Artifact specification

The Hugging Face dataset repository stores preprocessed relational graphs and
frozen text embeddings. Git stores only the downloader and this specification.

## Canonical layout

For holdout `rel-avito`, the canonical prefix is
`ra-holdout-rel-avito`. Its `train/` directory contains exactly
`rel-amazon`, `rel-f1`, `rel-hm`, `rel-stack`, and `rel-trial`; its `val/`
directory contains exactly `rel-avito`.

Each database directory contains:

| File | Purpose |
|---|---|
| `nodes.rkyv` | serialized relational cells and row metadata |
| `offsets.rkyv` | node offsets used by the sampler |
| `p2f_adj.rkyv` | parent-to-foreign adjacency with typed edges |
| `table_info.json` | split/table offsets and seed counts |
| `column_index.json` | table-column text indices |
| `relation_text_idxs.json` | typed relation indices into the text table |
| `text_emb_all-MiniLM-L12-v2.bin` | frozen bfloat16 text embeddings, width 384 |

`text.json` and `text_map.json` may be retained for provenance, but the public
installer does not download them because training does not read them.

## Installation contract

`data/download.py` resolves the requested HF revision to a commit, downloads
only runtime files, and checks:

- the holdout is absent from `train/`;
- all five expected training databases are present;
- every required file exists;
- relation indices are sorted, unique, and non-negative;
- relation indices address valid rows in the embedding file;
- existing local directories are never overwritten.

It then links each training database and the validation holdout into
`NANORT_PRE_ROOT` and writes a local provenance manifest. Both Python and Rust
honor `NANORT_PRE_ROOT`.

## Versioning

Use an immutable HF commit for a paper or leaderboard run. A moving branch such
as `main` is acceptable for exploration but insufficient evidence for a record.
Changing any graph, split, text embedding, or relation index requires a new HF
commit and invalidates direct timing comparison unless the benchmark version is
also advanced.

The artifact repository should publish checksums and byte sizes with each
versioned bundle.
