"""Lab 2 starter: parameter accounting for mBERT and CAMeLBERT."""

from transformers import AutoModel


def audit(checkpoint: str) -> dict:
    model = AutoModel.from_pretrained(checkpoint)

    buckets = {
      "embeddings": 0,
      "attention": 0,
      "ffn": 0,
      "norms": 0,
      "pooler": 0,
      "other": 0,
    }

    for name, parameter in model.named_parameters():
      count = parameter.numel()
      lower_name = name.lower()

      if "layernorm" in lower_name or "layer_norm" in lower_name:
          buckets["norms"] += count

      elif "embeddings" in lower_name:
          buckets["embeddings"] += count

      elif "attention" in lower_name:
         buckets["attention"] += count
  
      elif "intermediate" in lower_name or "output.dense" in lower_name:
          buckets["ffn"] += count

      elif "pooler" in lower_name:
          buckets["pooler"] += count

      else:
         buckets["other"] += count

    buckets["total"] = sum(
      value
      for key, value in buckets.items()
      if key != "total"
    )

    return buckets


if __name__ == "__main__":
    for ckpt in [
        "bert-base-multilingual-cased",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    ]:
        print(ckpt, audit(ckpt))
