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
| QA | span/null smoke | | | |

- Topic classifier validation improvement over baseline: `+0.2894` macro-F1.
- Topic classifier frozen-test improvement over baseline: `+0.5301` macro-F1.
- The required improvement threshold of `+0.08` was exceeded.
- The trained classifier artefact was saved to Google Drive.
- NER entity-level F1 target: `≥ 0.80`.
- Achieved frozen-test entity-level F1: `1.0000`.
- The trained NER artefact was saved to Google Drive.

## Lab 4 — Arabic model bake-off
| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| multilingual incumbent | | | | |
| Arabic dialect-aware | | | | |
| optional third model | | | | |

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | | | |
| + cross-encoder rerank | | | |
| cross-lingual slice | | | |

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
