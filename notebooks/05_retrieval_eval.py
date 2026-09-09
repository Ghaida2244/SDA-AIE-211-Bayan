"""Lab 5 starter: labelled-query retrieval evaluation."""

import json
import time
from pathlib import Path

import faiss
import numpy as np

from bayan.preprocessing.core import preprocess
from bayan.search.service import CaseSearch


QUERY_PATH = Path("data/search/bayan_queries.jsonl")
INDEX_PREFIX = "artifacts/case_index_eval"


def load_queries(path: Path):
    """Load labelled retrieval queries from JSON Lines."""

    queries = []

    with path.open("r", encoding="utf-8") as query_file:
        for line in query_file:
            line = line.strip()

            if line:
                queries.append(json.loads(line))

    return queries


def retrieve_stage_one(search_service, query: str, k: int = 10):
    """Retrieve cases with the bi-encoder without reranking."""

    normalized_query = preprocess(query)

    if not normalized_query:
        return [], []

    query_embedding = search_service.encoder.encode(
        [normalized_query],
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    faiss.normalize_L2(query_embedding)

    scores, indices = search_service.index.search(
        query_embedding,
        min(k, search_service.index.ntotal),
    )

    case_ids = [
        search_service.metadata[int(index)]["case_id"]
        for index in indices[0]
        if index >= 0
    ]

    valid_scores = [
        float(score)
        for score, index in zip(scores[0], indices[0])
        if index >= 0
    ]

    return case_ids, valid_scores


def ranking_metrics(retrieved_case_ids, relevant_case_ids):
    """Compute recall and reciprocal rank for one query."""

    relevant_set = set(relevant_case_ids)

    if not relevant_set:
        return 0.0, 0.0

    retrieved_relevant = [
        case_id
        for case_id in retrieved_case_ids
        if case_id in relevant_set
    ]

    recall = len(
        set(retrieved_relevant)
    ) / len(relevant_set)

    reciprocal_rank = 0.0

    for rank, case_id in enumerate(
        retrieved_case_ids,
        start=1,
    ):
        if case_id in relevant_set:
            reciprocal_rank = 1.0 / rank
            break

    return recall, reciprocal_rank


def summarize_rankings(rows):
    """Summarize retrieval metrics overall and by language."""

    summary = {}

    for slice_name, language in (
        ("all", None),
        ("arabic", "ar"),
        ("english", "en"),
    ):
        slice_rows = [
            row
            for row in rows
            if language is None or row["lang"] == language
        ]

        if not slice_rows:
            continue

        summary[slice_name] = {
            "queries": len(slice_rows),
            "recall_at_10": float(
                np.mean(
                    [row["recall_at_10"] for row in slice_rows]
                )
            ),
            "mrr_at_10": float(
                np.mean(
                    [row["reciprocal_rank"] for row in slice_rows]
                )
            ),
            "average_latency_ms": float(
                np.mean(
                    [row["latency_ms"] for row in slice_rows]
                )
            ),
        }

    return summary


def evaluate_answerable_queries(search_service, queries):
    """Evaluate answerable queries before and after reranking."""

    answerable_queries = [
        query
        for query in queries
        if not query["no_answer"]
    ]

    stage_one_rows = []
    reranked_rows = []

    for query in answerable_queries:
        stage_one_start = time.perf_counter()

        stage_one_ids, _ = retrieve_stage_one(
            search_service,
            query["query"],
            k=10,
        )

        stage_one_latency = (
            time.perf_counter() - stage_one_start
        ) * 1000

        stage_one_recall, stage_one_rr = ranking_metrics(
            stage_one_ids,
            query["relevant_case_ids"],
        )

        stage_one_rows.append(
            {
                "query_id": query["query_id"],
                "lang": query["lang"],
                "recall_at_10": stage_one_recall,
                "reciprocal_rank": stage_one_rr,
                "latency_ms": stage_one_latency,
            }
        )

        rerank_start = time.perf_counter()

        reranked_cases = search_service.search(
            query["query"],
            k=10,
            candidates=50,
            min_score=-1.0,
        )

        rerank_latency = (
            time.perf_counter() - rerank_start
        ) * 1000

        reranked_ids = [
            case["case_id"]
            for case in reranked_cases
        ]

        reranked_recall, reranked_rr = ranking_metrics(
            reranked_ids,
            query["relevant_case_ids"],
        )

        reranked_rows.append(
            {
                "query_id": query["query_id"],
                "lang": query["lang"],
                "recall_at_10": reranked_recall,
                "reciprocal_rank": reranked_rr,
                "latency_ms": rerank_latency,
            }
        )

    return {
        "stage_one": summarize_rankings(stage_one_rows),
        "reranked": summarize_rankings(reranked_rows),
    }



def tune_empty_result_threshold(search_service, queries):
    """Tune a cosine threshold for queries with no relevant case."""

    no_answer_queries = [
        query
        for query in queries
        if query["no_answer"]
    ]

    if not no_answer_queries:
        raise ValueError("No no-answer queries were found")

    maximum_scores = []

    for query in no_answer_queries:
        _, scores = retrieve_stage_one(
            search_service,
            query["query"],
            k=1,
        )

        maximum_scores.append(
            scores[0] if scores else -1.0
        )

    # Select the lowest threshold that rejects at least 85 percent
    required_empty = int(
        np.ceil(0.85 * len(maximum_scores))
    )

    sorted_scores = sorted(maximum_scores)
    boundary_score = sorted_scores[required_empty - 1]

    threshold = float(
        np.nextafter(boundary_score, np.inf)
    )

    empty_correct = sum(
        score < threshold
        for score in maximum_scores
    )

    answerable_queries = [
        query
        for query in queries
        if not query["no_answer"]
    ]

    answerable_kept = 0

    for query in answerable_queries:
        _, scores = retrieve_stage_one(
            search_service,
            query["query"],
            k=1,
        )

        if scores and scores[0] >= threshold:
            answerable_kept += 1

    return {
        "threshold": threshold,
        "empty_correct": int(empty_correct),
        "empty_total": len(no_answer_queries),
        "answerable_kept": int(answerable_kept),
        "answerable_total": len(answerable_queries),
    }


def main():
    """Run the complete bilingual retrieval evaluation."""

    print("Loading search models and index...")

    search_service = CaseSearch(INDEX_PREFIX)
    queries = load_queries(QUERY_PATH)

    print("Total queries:", len(queries))

    evaluation_results = evaluate_answerable_queries(
        search_service,
        queries,
    )

    threshold_results = tune_empty_result_threshold(
        search_service,
        queries,
    )

    stage_one = evaluation_results["stage_one"]
    reranked = evaluation_results["reranked"]

    evaluation_results["mrr_lift"] = (
        reranked["all"]["mrr_at_10"]
        - stage_one["all"]["mrr_at_10"]
    )

    evaluation_results["cross_lingual_gap"] = {
        "stage_one_recall_gap": abs(
            stage_one["arabic"]["recall_at_10"]
            - stage_one["english"]["recall_at_10"]
        ),
        "reranked_recall_gap": abs(
            reranked["arabic"]["recall_at_10"]
            - reranked["english"]["recall_at_10"]
        ),
        "stage_one_mrr_gap": abs(
            stage_one["arabic"]["mrr_at_10"]
            - stage_one["english"]["mrr_at_10"]
        ),
        "reranked_mrr_gap": abs(
            reranked["arabic"]["mrr_at_10"]
            - reranked["english"]["mrr_at_10"]
        ),
    }

    evaluation_results["empty_result"] = threshold_results

    output_path = Path(
        "artifacts/retrieval_eval_results.json"
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            evaluation_results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Stage-one results:")
    print(
        json.dumps(
            stage_one,
            indent=2,
        )
    )

    print()
    print("Reranked results:")
    print(
        json.dumps(
            reranked,
            indent=2,
        )
    )

    print()
    print(
        "MRR lift:",
        round(evaluation_results["mrr_lift"], 4),
    )
    print(
        "Cross-lingual gap:",
        evaluation_results["cross_lingual_gap"],
    )
    print(
        "Empty-result threshold:",
        threshold_results,
    )
    print(
        "Saved results:",
        output_path,
    )



if __name__ == "__main__":
    main()
