import os
from pathlib import Path

import numpy as np
import orjson
import strictfire
import torch
from ml_dtypes import bfloat16
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, batch_size, embedding_model, device_type):
        self.model = SentenceTransformer(
            f"sentence-transformers/{embedding_model}",
            device=device_type,
            model_kwargs={
                "dtype": torch.bfloat16 if device_type == "cuda" else torch.float32,
            },
        )
        self.model = torch.compile(self.model)
        self.batch_size = batch_size

    def __call__(self, text_list, device=None):
        chunks = []
        n = len(text_list)
        total_batches = (n + self.batch_size - 1) // self.batch_size
        print(
            f"[embed] encoding {n} texts in {total_batches} batches "
            f"(batch_size={self.batch_size})",
            flush=True,
        )
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_i = start // self.batch_size + 1
            if batch_i == 1 or batch_i % 100 == 0 or batch_i == total_batches:
                print(f"[embed] batch {batch_i}/{total_batches}", flush=True)
            chunks.append(
                self.model.encode(
                    text_list[start:end],
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    device=device,
                )
            )
        return np.concatenate(chunks, axis=0)


def main(
    dataset_name,
    device=None,
    batch_size=8192,
    embedding_model="all-MiniLM-L12-v2",
):
    if device is None:
        # Get list of all available CUDA devices
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            device = [f"cuda:{i}" for i in range(num_devices)]
            print(f"Found {num_devices} CUDA device(s): {device}")
        else:
            device = "cpu"
            print("No CUDA devices available, using CPU")

    if isinstance(device, list):
        device_type = torch.device(device[0]).type
    else:
        device_type = torch.device(device).type

    pre_root = Path(
        os.environ.get("NANORT_PRE_ROOT", Path.home() / ".cache" / "modded-nanort" / "pre")
    ).expanduser()
    text_path = pre_root / dataset_name / "text.json"
    with text_path.open() as f:
        raw = f.read()
    text_list = orjson.loads(raw)

    text_embedder = TextEmbedder(
        batch_size,
        embedding_model=embedding_model,
        device_type=device_type,
    )
    emb_list = text_embedder(text_list, device=device)

    emb_path = pre_root / dataset_name / f"text_emb_{embedding_model}.bin"
    emb = np.stack(emb_list).astype(bfloat16)
    emb.tofile(emb_path)


if __name__ == "__main__":
    strictfire.StrictFire(main)
