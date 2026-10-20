# Data artifacts

The training data lives in a Hugging Face dataset repository rather than in
Git. Each `ra-holdout-<database>` bundle fixes one of the six supported held-out
databases and contains the other five for training plus the holdout for
validation. For example:

```text
ra-holdout-rel-avito/
├── train/
│   ├── rel-amazon/
│   ├── rel-f1/
│   ├── rel-hm/
│   ├── rel-stack/
│   └── rel-trial/
└── val/
    └── rel-avito/
```

Every database directory includes the serialized graph, offsets, adjacency,
table and column metadata, frozen text embeddings, and
`relation_text_idxs.json`. The latter maps typed foreign-key descriptions into
the embedding table used by typed trilinear attention.

Install a pinned revision with:

```bash
pixi run python data/download.py \
  --repo-id GabrielVeloso/nanort-artifacts \
  --revision 8e6289edd0aa1d61b3b4c2ec1e1beed48224cd2c \
  --holdout rel-avito
```

The installer downloads only files consumed by training, verifies holdout
isolation and relation metadata, and creates links under
`~/.cache/modded-nanort/pre`. Override that location with
`NANORT_PRE_ROOT`.
