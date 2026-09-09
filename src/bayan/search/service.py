"""Provide two-stage bilingual semantic case search."""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from bayan.preprocessing.core import PREPROC_VERSION, preprocess


RERANKER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CaseSearch:
    """Load and search a versioned Bayan case index."""

    def __init__(
        self,
        prefix: str,
        reranker_model: str = RERANKER_NAME,
    ):
        """Load the index, metadata, encoder, and reranker."""

        index_path = Path(f"{prefix}_index.faiss")
        metadata_path = Path(f"{prefix}_metadata.json")
        manifest_path = Path(f"{prefix}_manifest.json")

        for required_path in (
            index_path,
            metadata_path,
            manifest_path,
        ):
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Missing search artifact: {required_path}"
                )

        self.manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        self.metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        self.index = faiss.read_index(str(index_path))

        required_manifest_keys = {
            "model",
            "preproc_version",
            "n_vectors",
            "dim",
        }

        missing_keys = required_manifest_keys.difference(
            self.manifest
        )

        if missing_keys:
            raise ValueError(
                f"Manifest is missing required keys: {sorted(missing_keys)}"
            )

        if self.manifest["preproc_version"] != PREPROC_VERSION:
            raise ValueError(
                "Index preprocessing version does not match the application"
            )

        if self.index.ntotal != self.manifest["n_vectors"]:
            raise ValueError(
                "Index vector count does not match the manifest"
            )

        if self.index.d != self.manifest["dim"]:
            raise ValueError(
                "Index dimension does not match the manifest"
            )

        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                "Metadata count does not match the index"
            )

        self.encoder = SentenceTransformer(
            self.manifest["model"]
        )
        self.reranker = CrossEncoder(reranker_model)

    def search(
        self,
        query: str,
        k: int = 5,
        candidates: int = 50,
        min_score: float = 0.25,
    ):
        """Retrieve candidates, rerank them, and return the best cases."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if k <= 0:
            raise ValueError("k must be greater than zero")

        if candidates <= 0:
            raise ValueError(
                "candidates must be greater than zero"
            )

        normalized_query = preprocess(query)

        if not normalized_query:
            return []

        query_embedding = self.encoder.encode(
            [normalized_query],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        # Match the normalization used when the index was built
        faiss.normalize_L2(query_embedding)

        candidate_count = min(
            max(k, candidates),
            self.index.ntotal,
        )

        retrieval_scores, retrieved_indices = self.index.search(
            query_embedding,
            candidate_count,
        )

        retrieved_cases = []

        for score, index_position in zip(
            retrieval_scores[0],
            retrieved_indices[0],
        ):
            if index_position < 0:
                continue

            if float(score) < min_score:
                continue

            case = dict(
                self.metadata[int(index_position)]
            )
            case["retrieval_score"] = float(score)
            retrieved_cases.append(case)

        # Return an empty result when no candidate passes the threshold
        if not retrieved_cases:
            return []

        reranker_pairs = [
            (
                normalized_query,
                preprocess(case["case_text"]),
            )
            for case in retrieved_cases
        ]

        reranker_scores = self.reranker.predict(
            reranker_pairs,
            show_progress_bar=False,
        )
        reranker_scores = np.asarray(
            reranker_scores,
            dtype=float,
        ).reshape(-1)

        for case, reranker_score in zip(
            retrieved_cases,
            reranker_scores,
        ):
            case["reranker_score"] = float(
                reranker_score
            )

        retrieved_cases.sort(
            key=lambda case: case["reranker_score"],
            reverse=True,
        )

        return retrieved_cases[:k]