"""Lab 7 starter: honest p50/p99 benchmark harness over production length mix."""


import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


DATA_PATH = Path("data/serving/bench_mix.npy")


def synchronize(device):
    """Wait for pending GPU operations before measuring time."""

    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(
    model,
    tokenizer,
    texts,
    *,
    device,
    padding,
    max_length,
    warmup=20,
):
    """Measure p50 and p99 latency for single-text inference."""

    if not texts:
        raise ValueError("texts must not be empty")

    warmup_count = min(warmup, len(texts))

    # Warm up the model before collecting latency measurements
    for text in texts[:warmup_count]:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=padding,
        )
        encoded = {
            name: tensor.to(device)
            for name, tensor in encoded.items()
        }

        with torch.inference_mode():
            model(**encoded)

    synchronize(device)
    latencies_ms = []

    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=padding,
        )
        encoded = {
            name: tensor.to(device)
            for name, tensor in encoded.items()
        }

        synchronize(device)
        start_time = time.perf_counter()

        with torch.inference_mode():
            model(**encoded)

        synchronize(device)

        latencies_ms.append(
            (time.perf_counter() - start_time) * 1000
        )

    return {
        "samples": len(latencies_ms),
        "p50_ms": float(
            np.percentile(latencies_ms, 50)
        ),
        "p99_ms": float(
            np.percentile(latencies_ms, 99)
        ),
        "mean_ms": float(
            np.mean(latencies_ms)
        ),
        "padding": padding,
        "max_length": int(max_length),
    }


def main():
    """Compare fixed and dynamic padding inference."""

    parser = argparse.ArgumentParser(
        description="Benchmark Bayan classifier inference."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the trained topic classifier.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of production-mix samples to benchmark.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output",
        default="artifacts/inference_benchmark.json",
    )
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("samples must be greater than zero")

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cpu":
        torch.set_num_threads(1)

    texts = np.load(
        DATA_PATH,
        allow_pickle=False,
    ).astype(str).tolist()
    texts = texts[: args.samples]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir
    )
    model.to(device)
    model.eval()

    token_lengths = [
        len(
            tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]
        )
        for text in texts
    ]

    recommended_max_length = max(
        8,
        int(np.percentile(token_lengths, 95)),
    )

    print("Device:", device)
    print("Samples:", len(texts))
    print(
        "Recommended p95 max length:",
        recommended_max_length,
    )

    baseline = benchmark(
        model,
        tokenizer,
        texts,
        device=device,
        padding="max_length",
        max_length=128,
    )

    dynamic_padding = benchmark(
        model,
        tokenizer,
        texts,
        device=device,
        padding=True,
        max_length=128,
    )

    dynamic_p95 = benchmark(
        model,
        tokenizer,
        texts,
        device=device,
        padding=True,
        max_length=recommended_max_length,
    )

    results = {
        "device": str(device),
        "baseline_fixed_128": baseline,
        "dynamic_padding_128": dynamic_padding,
        "dynamic_padding_p95": dynamic_p95,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print("Saved results:", output_path)


if __name__ == "__main__":
    main()