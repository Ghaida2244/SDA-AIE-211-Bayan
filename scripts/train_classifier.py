"""Lab 3A: fine-tune the Bayan topic classifier."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from bayan.models.data import build_topic_dataset


# Use the bilingual checkpoint selected in Lab 1
CHECKPOINT = "xlm-roberta-base"

# Keep the limit above the measured Lab 1 p95 sequence lengths
MAX_LENGTH = 128


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/topic_classifier",
        help=(
            "Where to save the trained classifier artefact "
            "(local path or mounted Drive path)."
        ),
    )

    return parser.parse_args()


def compute_metrics(eval_prediction):
    """Calculate macro-F1 and accuracy for model evaluation."""

    logits, labels = eval_prediction

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
    )

    accuracy = np.mean(
        predictions == labels
    )

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
    }


def main():
    """Train, evaluate, and save the Bayan topic classifier."""

    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load leakage-safe train, validation, and test splits
    dataset = build_topic_dataset()

    # Build deterministic label mappings from the training topics
    topic_names = sorted(
        set(dataset["train"]["topic"])
    )

    label_to_id = {
        topic: index
        for index, topic in enumerate(topic_names)
    }

    id_to_label = {
        index: topic
        for topic, index in label_to_id.items()
    }

    print("Topics:", topic_names)
    print("Number of topics:", len(topic_names))

    # Load the tokenizer selected from the Lab 1 measurements
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT
    )

    def tokenize_batch(batch):
        """Tokenize text and convert topic names into numeric labels."""

        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

        encoded["labels"] = [
            label_to_id[topic]
            for topic in batch["topic"]
        ]

        return encoded

    # Tokenize all splits and remove unused source columns
    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Load XLM-R with a new classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(topic_names),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    # Dynamically pad each batch to its longest sequence
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    # Configure training, evaluation, and checkpoint saving
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
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=42,
    )

    # Create the Hugging Face training controller
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    # Fine-tune the classifier and measure wall-clock time
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

    # Save the reusable model and tokenizer
    trainer.save_model(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    trainer.save_state()

    # Save measured metrics with the model artefact
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

    print("\nTraining completed")
    print(
        f"Training time: {training_time:.2f} seconds"
    )
    print(
        "Validation macro-F1:",
        validation_metrics["validation_macro_f1"],
    )
    print(
        "Frozen test macro-F1:",
        test_metrics["test_macro_f1"],
    )
    print(
        "Saved model:",
        output_dir,
    )


if __name__ == "__main__":
    main()