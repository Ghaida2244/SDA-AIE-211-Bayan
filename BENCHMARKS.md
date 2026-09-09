# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit
| Tokenizer | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |
|---|---:|---:|---:|---:|---:|
| mBERT | 2.15 | 1.51 | 27 | 25 | 0.005 |
| XLM-R | 1.67 | 1.43 | 21 | 23 | 0.000 |
| CAMeLBERT | 1.41 | 2.70 | 20 | 38 | 0.008 |
| DistilBERT | 4.53 | 1.30 | 47 | 21 | 0.002 |

- Golden preprocessing: 25 / 25 passed
- PII masking recall: 60  / 60 = 100%

## Lab 3 — Models
| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | 0.7106 | 0.4699 | 0.1455 s |
| Topic classifier (XLM-R) | macro-F1 | 1.0000 | 1.0000 | 276.79 s |
| NER (XLM-R) | entity-F1 | 1.0000 | 1.0000 | 170.02 s |
| QA (XLM-R SQuAD2) | span/null smoke | 9/9 answerable | 3/3 null | Pretrained |

- Topic classifier validation improvement over baseline: `+0.2894` macro-F1.
- Topic classifier frozen-test improvement over baseline: `+0.5301` macro-F1.
- The required improvement threshold of `+0.08` was exceeded.
- The trained classifier artefact was saved to Google Drive.
- NER entity-level F1 target: `≥ 0.80`.
- Achieved frozen-test entity-level F1: `1.0000`.
- The trained NER artefact was saved to Google Drive.
- QA checkpoint: `deepset/xlm-roberta-base-squad2`.
- Answerable QA smoke result: `9/9` correct spans.
- Unanswerable QA smoke result: `3/3` returned `answer=None`.
- SQuAD-style answer normalization was used for fair exact-match evaluation.

## Lab 4: Arabic model bake-off

| Model | All macro-F1 | Gulf macro-F1 | MSA macro-F1 | Training time |
|---|---:|---:|---:|---:|
| CAMeLBERT-mix | 1.0000 | 1.0000 | 1.0000 | 102.60 seconds |
| CAMeLBERT-DA | 1.0000 | 1.0000 | 1.0000 | 121.19 seconds |

Both models achieved perfect macro-F1 on all test slices. CAMeLBERT-mix was selected because it produced the same performance in less training time and supports both MSA and dialectal Arabic.

The expected four-point Gulf improvement could not be measured because both models reached the maximum score. This indicates a ceiling effect in the synthetic dataset.

## Lab 5: Bilingual semantic search

| Stage | Recall@10 | MRR@10 | Average latency |
|---|---:|---:|---:|
| Bi-encoder | 0.0590 | 0.0605 | 13.80 ms |
| Cross-encoder reranking | 0.0641 | 0.0590 | 58.49 ms |

| Reranked slice | Recall@10 | MRR@10 |
|---|---:|---:|
| Arabic | 0.0722 | 0.0530 |
| English | 0.0571 | 0.0641 |

The tuned cosine threshold was 0.4586. It correctly returned an empty result for 20/20 no-answer queries while keeping all 130 answerable queries.

- no-answer empty-correct: ___ / 20
- cross-lingual gap: ___

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | | |
| dialect-aware | | | | |

- paired comparison verdict:
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Optimisation ladder
| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | | | | |
| fp32 torch @128 dynamic | | | | |
| ONNX fp32 @128 | | | | |
| ONNX INT8 @128 | | | | |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:
