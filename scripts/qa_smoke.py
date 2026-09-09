"""Lab 3B: run the answerable and unanswerable QA smoke tests."""


import re
import string
import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)

from bayan.models.qa import best_span


# Use a multilingual extractive-QA checkpoint with no-answer support
CHECKPOINT = "deepset/xlm-roberta-base-squad2"

SMOKE_PATH = Path(
    "data/eval/qa_smoke_set.json"
)

QA_DATA_PATH = Path(
    "data/models/bayan_qa.json"
)

NULL_THRESHOLD = 0.0
MAX_LENGTH = 384


def load_questions(path: Path):
    """Load QA examples from a SQuAD-style JSON file."""

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        payload = json.load(file)

    examples = []

    for document in payload["data"]:
        for paragraph in document["paragraphs"]:
            context = paragraph["context"]

            for question in paragraph["qas"]:
                examples.append({
                    "id": question["id"],
                    "question": question["question"],
                    "context": context,
                    "answers": question["answers"],
                    "is_impossible": question["is_impossible"],
                })

    return examples


def normalize_answer(text):
    """Apply standard SQuAD-style normalization for exact matching."""

    text = text.casefold()

    # Remove punctuation
    text = "".join(
        character
        for character in text
        if character not in string.punctuation
    )

    # Remove common English articles
    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    # Collapse repeated whitespace
    return " ".join(text.split())


def predict_answer(example, tokenizer, model, device):
    """Predict one answer span with honest no-answer handling."""

    encoded = tokenizer(
        example["question"],
        example["context"],
        return_offsets_mapping=True,
        truncation="only_second",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # Preserve offsets before sending tensors to the model
    offset_mapping = encoded.pop(
        "offset_mapping"
    )[0].tolist()

    sequence_ids = encoded.sequence_ids(
        batch_index=0
    )

    # Keep offsets only for context tokens
    context_offsets = [
        tuple(offset)
        if sequence_id == 1
        else None
        for offset, sequence_id in zip(
            offset_mapping,
            sequence_ids,
        )
    ]

    model_inputs = {
        name: tensor.to(device)
        for name, tensor in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

    start_logits = (
        outputs.start_logits[0]
        .detach()
        .cpu()
        .numpy()
    )

    end_logits = (
        outputs.end_logits[0]
        .detach()
        .cpu()
        .numpy()
    )

    # The CLS position represents the no-answer candidate
    null_score = float(
        start_logits[0] + end_logits[0]
    )

    result = best_span(
        start_logits,
        end_logits,
        context_offsets,
        null_score=null_score,
        null_threshold=NULL_THRESHOLD,
    )

    if result["answer"] is None:
        return None

    return example["context"][
        result["start"]:result["end"]
    ]


def main():
    """Run nine answerable and three unanswerable QA checks."""

    smoke_examples = load_questions(
        SMOKE_PATH
    )

    full_qa_examples = load_questions(
        QA_DATA_PATH
    )

    # Use nine answerable examples from the smoke fixture
    answerable_examples = [
        example
        for example in smoke_examples
        if not example["is_impossible"]
    ][:9]

    # Supplement the fixture with three supplied null examples
    unanswerable_examples = [
        example
        for example in full_qa_examples
        if example["is_impossible"]
    ][:3]

    if len(answerable_examples) != 9:
        raise ValueError(
            "Expected nine answerable smoke examples"
        )

    if len(unanswerable_examples) != 3:
        raise ValueError(
            "Expected three unanswerable smoke examples"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("QA checkpoint:", CHECKPOINT)

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT,
        use_fast=True,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        CHECKPOINT
    )

    model.to(device)
    model.eval()

    answerable_correct = 0
    null_correct = 0

    print("\nAnswerable examples:")

    for example in answerable_examples:
        predicted_answer = predict_answer(
            example,
            tokenizer,
            model,
            device,
        )

        expected_answer = example["answers"][0]["text"]

        is_correct = (
            predicted_answer is not None
            and normalize_answer(predicted_answer)
            == normalize_answer(expected_answer)
        )

        answerable_correct += int(is_correct)

        print(
            example["id"],
            "| predicted:",
            repr(predicted_answer),
            "| expected:",
            repr(expected_answer),
            "| correct:",
            is_correct,
        )

    print("\nUnanswerable examples:")

    for example in unanswerable_examples:
        predicted_answer = predict_answer(
            example,
            tokenizer,
            model,
            device,
        )

        is_correct = predicted_answer is None
        null_correct += int(is_correct)

        print(
            example["id"],
            "| predicted:",
            repr(predicted_answer),
            "| expected: None",
            "| correct:",
            is_correct,
        )

    print(
        f"\nAnswerable correct: "
        f"{answerable_correct}/9"
    )

    print(
        f"Unanswerable correct: "
        f"{null_correct}/3"
    )

    if answerable_correct != 9:
        raise AssertionError(
            "QA smoke test failed on answerable examples"
        )

    if null_correct != 3:
        raise AssertionError(
            "QA smoke test failed on unanswerable examples"
        )

    print("QA smoke test passed")



if __name__ == "__main__":
    main()