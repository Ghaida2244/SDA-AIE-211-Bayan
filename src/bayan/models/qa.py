"""Lab 3B: extractive QA post-processing."""

import numpy as np


def best_span(
    start_logits,
    end_logits,
    offsets,
    *,
    null_score,
    null_threshold,
    max_answer_len=30,
    top_k=20,
):
    """Select the best valid answer span or return an honest null answer."""

    start_logits = np.asarray(start_logits)
    end_logits = np.asarray(end_logits)

    if len(start_logits) != len(end_logits):
        raise ValueError(
            "Start and end logits must have the same length"
        )

    if len(start_logits) != len(offsets):
        raise ValueError(
            "Logits and offsets must have the same length"
        )

    # Keep only the strongest candidate start and end positions
    start_candidates = np.argsort(
        start_logits
    )[-top_k:][::-1]

    end_candidates = np.argsort(
        end_logits
    )[-top_k:][::-1]

    best_score = float("-inf")
    best_start = None
    best_end = None

    for start_index in start_candidates:
        for end_index in end_candidates:
            # Ignore special tokens and tokens outside the context
            if offsets[start_index] is None:
                continue

            if offsets[end_index] is None:
                continue

            # Reject spans whose end appears before their start
            if end_index < start_index:
                continue

            # Reject answers that exceed the allowed token length
            answer_length = (
                end_index - start_index + 1
            )

            if answer_length > max_answer_len:
                continue

            start_character = offsets[start_index][0]
            end_character = offsets[end_index][1]

            # Reject empty or inverted character spans
            if end_character <= start_character:
                continue

            span_score = float(
                start_logits[start_index]
                + end_logits[end_index]
            )

            if span_score > best_score:
                best_score = span_score
                best_start = int(start_character)
                best_end = int(end_character)

    # Return null when no valid answer span exists
    if best_start is None:
        return {
            "answer": None,
            "score": None,
            "null_score": float(null_score),
        }

    # Prefer null when it is sufficiently stronger than the best span
    null_difference = (
        float(null_score) - best_score
    )

    if null_difference > null_threshold:
        return {
            "answer": None,
            "score": best_score,
            "null_score": float(null_score),
        }

    return {
        "answer": (best_start, best_end),
        "start": best_start,
        "end": best_end,
        "score": best_score,
        "null_score": float(null_score),
    }