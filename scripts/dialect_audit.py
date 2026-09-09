"""Lab 4: audit the dialect distribution in Bayan Arabic data."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/raw/bayan_feedback.csv"
)


def main():
    """Print the dialect distribution for Arabic feedback."""

    dataframe = pd.read_csv(
        DATA_PATH
    )

    # Keep only Arabic feedback
    arabic_data = dataframe[
        dataframe["lang"] == "ar"
    ].copy()

    # Count examples in each dialect region
    dialect_counts = (
        arabic_data["dialect_region"]
        .value_counts(dropna=False)
        .rename_axis("dialect_region")
        .reset_index(name="count")
    )

    total_arabic = len(arabic_data)

    dialect_counts["percentage"] = (
        dialect_counts["count"]
        / total_arabic
        * 100
    )

    print("Total Arabic examples:", total_arabic)
    print("\nDialect distribution:")

    print(
        dialect_counts.to_string(
            index=False,
            formatters={
                "percentage": lambda value: f"{value:.2f}%"
            },
        )
    )

    print(
        "\nImplication: Evaluating only on MSA would not "
        "represent most Bayan Arabic users because the "
        "majority of the Arabic data is Gulf dialect."
    )


if __name__ == "__main__":
    main()