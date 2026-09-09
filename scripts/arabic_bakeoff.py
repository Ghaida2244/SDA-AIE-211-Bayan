"""Compare Arabic-focused checkpoints on Bayan topic classification."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


DATA_PATH = Path("data/raw/bayan_feedback.csv")

CHECKPOINTS = {
    "CAMeLBERT-mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "CAMeLBERT-DA": "CAMeL-Lab/bert-base-arabic-camelbert-da",
}

def build_arabic_dataset():
    """Create leakage-safe Arabic train, validation, and test splits."""

    dataframe = pd.read_csv(DATA_PATH)

    # Keep the columns required for training, slicing, and grouped splitting
    dataframe = dataframe.loc[
        dataframe["lang"].eq("ar"),
        [
            "text",
            "topic",
            "dialect_region",
            "citizen_group_id",
        ],
    ].copy()

    topic_names = sorted(dataframe["topic"].unique())
    label_to_id = {
        topic: index
        for index, topic in enumerate(topic_names)
    }

    dataframe["labels"] = dataframe["topic"].map(label_to_id)

    # Reserve 30 percent of citizen groups for validation and testing
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=42,
    )

    train_indices, temporary_indices = next(
        first_split.split(
            dataframe,
            groups=dataframe["citizen_group_id"],
        )
    )

    train_frame = dataframe.iloc[train_indices].copy()
    temporary_frame = dataframe.iloc[temporary_indices].copy()

    # Divide the reserved citizen groups equally between validation and test
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42,
    )

    validation_indices, test_indices = next(
        second_split.split(
            temporary_frame,
            groups=temporary_frame["citizen_group_id"],
        )
    )

    validation_frame = temporary_frame.iloc[validation_indices].copy()
    test_frame = temporary_frame.iloc[test_indices].copy()

    dataset_splits = {}

    for split_name, split_frame in {
        "train": train_frame,
        "validation": validation_frame,
        "test": test_frame,
    }.items():
        split_frame = split_frame[
            ["text", "labels", "dialect_region"]
        ].reset_index(drop=True)

        dataset_splits[split_name] = Dataset.from_pandas(
            split_frame,
            preserve_index=False,
        )

    return DatasetDict(dataset_splits), topic_names


def compute_metrics(evaluation_prediction):
    """Compute overall macro-F1 and accuracy during validation."""

    logits, labels = evaluation_prediction
    predictions = np.argmax(logits, axis=-1)

    return {
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "accuracy": accuracy_score(labels, predictions),
    }


def compute_slice_metrics(labels, predictions, dialect_regions):
    """Compute overall, Gulf, and MSA macro-F1 scores."""

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    dialect_regions = np.asarray(dialect_regions)

    results = {
        "all_macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    }

    for region in ("Gulf", "MSA"):
        region_mask = dialect_regions == region

        results[f"{region.lower()}_macro_f1"] = f1_score(
            labels[region_mask],
            predictions[region_mask],
            average="macro",
            zero_division=0,
        )

        results[f"{region.lower()}_examples"] = int(
            region_mask.sum()
        )

    return results


def train_checkpoint(
    model_name,
    checkpoint,
    dataset,
    topic_names,
    output_root,
):
    """Train one checkpoint and evaluate Arabic dialect slices."""

    print()
    print("=" * 60)
    print("Training:", model_name)
    print("Checkpoint:", checkpoint)

    label_to_id = {
        topic: index
        for index, topic in enumerate(topic_names)
    }
    id_to_label = {
        index: topic
        for topic, index in label_to_id.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        use_fast=True,
    )

    def tokenize_batch(batch):
        """Tokenize one batch and preserve its topic labels."""

        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )
        encoded["labels"] = batch["labels"]
        return encoded

    # Keep the dialect values separately for slice evaluation
    test_dialects = dataset["test"]["dialect_region"]

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(topic_names),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    model_output_dir = output_root / model_name.lower().replace(
        "-",
        "_",
    )
    model_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
        ),
        compute_metrics=compute_metrics,
    )

    start_time = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - start_time

    test_output = trainer.predict(
        tokenized_dataset["test"]
    )
    test_predictions = np.argmax(
        test_output.predictions,
        axis=-1,
    )

    results = compute_slice_metrics(
        test_output.label_ids,
        test_predictions,
        test_dialects,
    )
    results["training_seconds"] = training_seconds
    results["checkpoint"] = checkpoint

    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))

    print("Results:", results)
    print("Saved model:", model_output_dir)

    return results


def main():
    """Run the Arabic checkpoint bake-off."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare Arabic checkpoints on Bayan topic classification."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/arabic_bakeoff",
        help="Directory used to save models and evaluation results.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    dataset, topic_names = build_arabic_dataset()

    print("Topics:", topic_names)
    print("Train examples:", len(dataset["train"]))
    print("Validation examples:", len(dataset["validation"]))
    print("Test examples:", len(dataset["test"]))

    all_results = {}

    for model_name, checkpoint in CHECKPOINTS.items():
        all_results[model_name] = train_checkpoint(
            model_name=model_name,
            checkpoint=checkpoint,
            dataset=dataset,
            topic_names=topic_names,
            output_root=output_root,
        )

    results_path = output_root / "results.json"

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as results_file:
        json.dump(
            all_results,
            results_file,
            ensure_ascii=False,
            indent=2,
            default=float,
        )

    print()
    print("=" * 60)
    print("Arabic model comparison")

    for model_name, results in all_results.items():
        print(
            model_name,
            "| All:",
            f"{results['all_macro_f1']:.4f}",
            "| Gulf:",
            f"{results['gulf_macro_f1']:.4f}",
            "| MSA:",
            f"{results['msa_macro_f1']:.4f}",
            "| Seconds:",
            f"{results['training_seconds']:.2f}",
        )

    print("Saved results:", results_path)

if __name__ == "__main__":
    main()
