"""Lab 3B starter: fine-tune token classification with correct alignment."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score as entity_f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from bayan.models.ner import align_labels


# Use the bilingual checkpoint selected in Lab 1
CHECKPOINT = "xlm-roberta-base"
# Use the supplied word-level CoNLL training dataset
DATA_PATH = Path("data/models/bayan_ner.conll")
# Keep the token limit comfortably above the short Bayan sequences
MAX_LENGTH = 128
# Use deterministic train, validation, and test splits
RANDOM_SEED = 42

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/ner",
        help="Where to save the trained NER artefact (local path or mounted Drive path).",
    )
    return parser.parse_args()


def read_conll(path: Path):
    """Read word-level tokens and BIO labels from a CoNLL file."""

    sentences = []
    sentence_labels = []

    current_tokens = []
    current_labels = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            # An empty line marks the end of one sentence
            if not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    sentence_labels.append(current_labels)

                    current_tokens = []
                    current_labels = []

                continue

            parts = line.split()

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid CoNLL row at line {line_number}: {line}"
                )

            token, label = parts

            current_tokens.append(token)
            current_labels.append(label)

    # Preserve the final sentence when the file has no trailing blank line
    if current_tokens:
        sentences.append(current_tokens)
        sentence_labels.append(current_labels)

    return sentences, sentence_labels


def build_compute_metrics(id_to_label):
    """Create an entity-level NER evaluation function."""

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction

        predictions = np.argmax(
            logits,
            axis=-1,
        )

        true_predictions = []
        true_labels = []

        for prediction_row, label_row in zip(
            predictions,
            labels,
        ):
            filtered_predictions = []
            filtered_labels = []

            for prediction_id, label_id in zip(
                prediction_row,
                label_row,
            ):
                # Ignore special tokens and non-first subword pieces
                if label_id == -100:
                    continue

                filtered_predictions.append(
                    id_to_label[int(prediction_id)]
                )

                filtered_labels.append(
                    id_to_label[int(label_id)]
                )

            true_predictions.append(
                filtered_predictions
            )

            true_labels.append(
                filtered_labels
            )

        return {
            "entity_f1": entity_f1_score(
                true_labels,
                true_predictions,
            ),
            "token_accuracy": accuracy_score(
                true_labels,
                true_predictions,
            ),
        }

    return compute_metrics


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the supplied CoNLL sentences and word-level labels
sentences, sentence_labels = read_conll(DATA_PATH)

# Build deterministic label mappings
label_names = sorted({
    label
    for labels in sentence_labels
    for label in labels
})

label_to_id = {
    label: index
    for index, label in enumerate(label_names)
}

id_to_label = {
    index: label
    for label, index in label_to_id.items()
}

print("Number of sentences:", len(sentences))
print("NER labels:", label_names)

# Convert string labels into numeric model labels
numeric_labels = [
    [
        label_to_id[label]
        for label in labels
    ]
    for labels in sentence_labels
]

# Create a Hugging Face dataset
full_dataset = Dataset.from_dict({
    "tokens": sentences,
    "ner_tags": numeric_labels,
})

# Create deterministic 80% train, 10% validation, and 10% test splits
train_and_remaining = full_dataset.train_test_split(
    test_size=0.20,
    seed=RANDOM_SEED,
)

validation_and_test = train_and_remaining[
    "test"
].train_test_split(
    test_size=0.50,
    seed=RANDOM_SEED,
)

dataset = {
    "train": train_and_remaining["train"],
    "validation": validation_and_test["train"],
    "test": validation_and_test["test"],
}

print("Train sentences:", len(dataset["train"]))
print("Validation sentences:", len(dataset["validation"]))
print("Test sentences:", len(dataset["test"]))

# Load the tokenizer selected in Lab 1
tokenizer = AutoTokenizer.from_pretrained(
    CHECKPOINT,
    use_fast=True,
)


def tokenize_and_align(batch):
    """Tokenize pre-split words and align their NER labels."""

    tokenized = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    aligned_batch_labels = []

    for batch_index, word_labels in enumerate(
        batch["ner_tags"]
    ):
        word_ids = tokenized.word_ids(
            batch_index=batch_index
        )

        aligned_batch_labels.append(
            align_labels(
                word_ids,
                word_labels,
            )
        )

    tokenized["labels"] = aligned_batch_labels

    return tokenized


# Tokenize and align all dataset splits
tokenized_dataset = {
    split_name: split_dataset.map(
        tokenize_and_align,
        batched=True,
        remove_columns=split_dataset.column_names,
    )
    for split_name, split_dataset in dataset.items()
}



# Load XLM-R with a token-classification head
model = AutoModelForTokenClassification.from_pretrained(
    CHECKPOINT,
    num_labels=len(label_names),
    id2label=id_to_label,
    label2id=label_to_id,
)

# Dynamically pad tokens and aligned labels within each batch
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
)

# Configure NER fine-tuning and checkpoint selection
training_args = TrainingArguments(
    output_dir=str(output_dir),
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="entity_f1",
    greater_is_better=True,
    save_total_limit=1,
    logging_steps=50,
    report_to="none",
    fp16=torch.cuda.is_available(),
    seed=RANDOM_SEED,
)



# Create the Hugging Face NER training controller
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=build_compute_metrics(id_to_label),
    processing_class=tokenizer,
)

# Fine-tune the NER model and measure wall-clock time
training_start = time.perf_counter()

train_result = trainer.train()

training_time = (
    time.perf_counter() - training_start
)

# Evaluate the best checkpoint on validation data
validation_metrics = trainer.evaluate(
    tokenized_dataset["validation"],
    metric_key_prefix="validation",
)

# Evaluate once on the frozen test split
test_metrics = trainer.evaluate(
    tokenized_dataset["test"],
    metric_key_prefix="test",
)

# Save the reusable NER model and tokenizer
trainer.save_model(
    str(output_dir)
)

tokenizer.save_pretrained(
    str(output_dir)
)

trainer.save_state()

# Save all measured metrics with the model artefact
train_metrics = dict(
    train_result.metrics
)

train_metrics["wall_clock_seconds"] = (
    training_time
)

trainer.save_metrics(
    "train",
    train_metrics,
)

trainer.save_metrics(
    "validation",
    validation_metrics,
)

trainer.save_metrics(
    "test",
    test_metrics,
)

print("\nNER training completed")

print(
    f"Training time: {training_time:.2f} seconds"
)

print(
    "Validation entity-level F1:",
    validation_metrics["validation_entity_f1"],
)

print(
    "Frozen test entity-level F1:",
    test_metrics["test_entity_f1"],
)

print(
    "Saved NER model:",
    output_dir,
)



if __name__ == "__main__":
    main()
