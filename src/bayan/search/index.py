"""Lab 5 starter: versioned FAISS index build."""

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from bayan.preprocessing.core import PREPROC_VERSION, preprocess


DATA_PATH = Path("data/search/bayan_cases.csv")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_index(
    prefix: str,
    limit: int | None = None,
    model_name: str = MODEL_NAME,
):
    """Build and persist a normalized FAISS index and its metadata."""

    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(DATA_PATH)

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        dataframe = dataframe.head(limit)

    if dataframe.empty:
        raise ValueError("No cases are available for indexing")

   # Combine the case description and resolution for richer retrieval
    normalized_texts = [
    preprocess(f"{case_text} {resolution}")
    for case_text, resolution in zip(
        dataframe["case_text"].fillna("").astype(str),
        dataframe["resolution"].fillna("").astype(str),
    )
    ]

    encoder = SentenceTransformer(model_name)

    embeddings = encoder.encode(
        normalized_texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    # L2 normalization converts inner-product search into cosine search
    faiss.normalize_L2(embeddings)

    dimension = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_path = Path(f"{prefix}_index.faiss")
    metadata_path = Path(f"{prefix}_metadata.json")
    manifest_path = Path(f"{prefix}_manifest.json")

    faiss.write_index(
        index,
        str(index_path),
    )

    metadata_columns = [
        "case_id",
        "lang",
        "topic",
        "case_text",
        "resolution",
        "status",
    ]

    metadata = dataframe[
        metadata_columns
    ].fillna("").to_dict(orient="records")

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "model": model_name,
        "preproc_version": PREPROC_VERSION,
        "n_vectors": int(index.ntotal),
        "dim": dimension,
        "metric": "cosine",
        "normalized": True,
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return manifest