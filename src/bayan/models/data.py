"""Lab 3 starter: dataset construction and split integrity."""

from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

DATA_PATH = Path("data/raw/bayan_feedback.csv")


def build_topic_dataset(data_path=DATA_PATH):
  """Build leakage-safe train, validation, and test datasets."""
  dataframe = pd.read_csv(data_path)

  required_columns = {
        "text",
        "topic",
        "citizen_group_id",
        "split",
      }

  missing_columns = required_columns - set(dataframe.columns)

  if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )
  
  datasets = {}

  for split_name in ["train", "validation", "test"]:
        split_dataframe = dataframe[
            dataframe["split"] == split_name
        ].reset_index(drop=True)

        datasets[split_name] = Dataset.from_pandas(
            split_dataframe,
            preserve_index=False,
        )

  dataset_dict = DatasetDict(datasets)

  # Verify that citizen groups do not overlap across dataset splits

  train_ids = set(dataset_dict["train"]["citizen_group_id"])
  validation_ids = set(
        dataset_dict["validation"]["citizen_group_id"]
    )
  test_ids = set(dataset_dict["test"]["citizen_group_id"])

  if not train_ids.isdisjoint(validation_ids):
        raise ValueError(
            "Citizen overlap detected between train and validation"
        )

  if not train_ids.isdisjoint(test_ids):
        raise ValueError(
            "Citizen overlap detected between train and test"
        )
  if not validation_ids.isdisjoint(test_ids):
        raise ValueError(
            "Citizen overlap detected between validation and test"
        )

  return dataset_dict








