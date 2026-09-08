"""Lab 1 starter: audit four tokenizer candidates on Bayan AR/EN text."""
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

DATA = Path("data/raw/bayan_feedback.csv")


def fertility(tokenizer, texts) -> float:
    words = 0
    pieces = 0

    for text in texts:
        text = str(text)

        words += len(text.split())
        pieces += len(tokenizer.tokenize(text))

    return pieces / max(words, 1)


def main():
    # قراءة بيانات Bayan
    corpus = pd.read_csv(DATA)

    # فصل النصوص العربية عن الإنجليزية
    ar_texts = (
      corpus[corpus["lang"] == "ar"]["text"]
     .dropna()
    .tolist()
    )

    en_texts = (
      corpus[corpus["lang"] == "en"]["text"]
      .dropna()
      .tolist()
    )

    print(f"Arabic rows: {len(ar_texts)}")
    print(f"English rows: {len(en_texts)}")
    print()

    for checkpoint, label in CANDIDATES.items():
      print("=" * 60)
      print(label)
      print(checkpoint)

      tokenizer = AutoTokenizer.from_pretrained(checkpoint)

      # حساب Fertility للعربية والإنجليزية
      ar_fertility = fertility(tokenizer, ar_texts)
      en_fertility = fertility(tokenizer, en_texts)

      # حساب طول كل نص بعد تحويله إلى Tokens
      ar_lengths = [
            len(tokenizer.encode(text, add_special_tokens=True))
            for text in ar_texts
        ]

      en_lengths = [
         len(tokenizer.encode(text, add_special_tokens=True))
         for text in en_texts
        ]

      # حساب الطول الذي تقع تحته 95% من النصوص
      ar_p95 = np.percentile(ar_lengths, 95)
      en_p95 = np.percentile(en_lengths, 95)

      # حساب نسبة الرموز العربية غير المعروفة
      ar_token_ids = [
         tokenizer.encode(text, add_special_tokens=False)
         for text in ar_texts
       ]

      ar_token_count = sum(len(ids) for ids in ar_token_ids)

      if tokenizer.unk_token_id is None:
                ar_unk_count = 0
      else:
            ar_unk_count = sum(
               ids.count(tokenizer.unk_token_id)
               for ids in ar_token_ids
            )

      ar_unk_rate = ar_unk_count / max(ar_token_count, 1)

      print(f"AR fertility: {ar_fertility:.2f}")
      print(f"EN fertility: {en_fertility:.2f}")
      print(f"AR p95 length: {ar_p95:.0f} tokens")
      print(f"EN p95 length: {en_p95:.0f} tokens")
      print(f"AR UNK rate: {ar_unk_rate:.3f}")
      print()

if __name__ == "__main__":
    main()
