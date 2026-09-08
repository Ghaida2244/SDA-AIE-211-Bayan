"""Lab 3A: TF-IDF and LinearSVC topic-classification baseline."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from bayan.models.data import build_topic_dataset

import time

def main():
    # Load the leakage-safe dataset splits
    dataset = build_topic_dataset()

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Use a deliberately small vocabulary for the lightweight baseline
    vectorizer = TfidfVectorizer(
        max_features=20,
    )

    train_features = vectorizer.fit_transform(
        train_dataset["text"]
    )

    validation_features = vectorizer.transform(
        validation_dataset["text"]
    )

    test_features = vectorizer.transform(
        test_dataset["text"]
    )


    classifier = LinearSVC()

   # Measure classifier training time
    start_time = time.perf_counter()

    classifier.fit(
    train_features,
    train_dataset["topic"],
    )

    train_time = time.perf_counter() - start_time

    # Evaluate on validation and frozen test splits
    validation_predictions = classifier.predict(
        validation_features
    )

    test_predictions = classifier.predict(
        test_features
    )

    validation_f1 = f1_score(
        validation_dataset["topic"],
        validation_predictions,
        average="macro",
    )

    test_f1 = f1_score(
        test_dataset["topic"],
        test_predictions,
        average="macro",
    )

    print(f"Validation macro-F1: {validation_f1:.4f}")
    print(f"Frozen test macro-F1: {test_f1:.4f}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Training time: {train_time:.4f} seconds")


if __name__ == "__main__":
    main()