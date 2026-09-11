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
- PII masking recall: 60 / 60 = 100%
- Selected tokenizer: XLM-R
- Reason: XLM-R worked well for both Arabic and English and had a 0.000 Arabic UNK rate.

## Lab 3 — Models

| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | 0.7106 | 0.4699 | 0.15 seconds |
| XLM-R topic classifier | macro-F1 | 1.0000 | 1.0000 | 276.79 seconds |
| XLM-R NER | entity-F1 | 1.0000 | 1.0000 | 170.02 seconds |
| QA | span/null smoke | 12 / 12 | 12 / 12 | Not trained |

- The TF-IDF model was used as a simple baseline.
- The XLM-R topic classifier achieved the best topic classification result.
- The NER model correctly identified the entities in the test data.
- The QA model passed 9 answerable examples and 3 no-answer examples.

## Lab 4 — Arabic model bake-off

| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| CAMeLBERT-mix | 1.0000 | 1.0000 | 1.0000 | 1.41 |
| CAMeLBERT-DA | 1.0000 | 1.0000 | 1.0000 | Not measured |
| Optional third model | Not run | Not run | Not run | Not run |

- CAMeLBERT-mix training time: 102.60 seconds
- CAMeLBERT-DA training time: 121.19 seconds
- Selected model: CAMeLBERT-mix
- Reason: Both models had the same score, but CAMeLBERT-mix was faster.
- The expected Gulf improvement was not found because both models reached the maximum score.
- This result shows a ceiling effect in the synthetic dataset.

### NER segmentation

| Configuration | Validation entity-F1 | Frozen test entity-F1 | Train time |
|---|---:|---:|---:|
| Without clitic segmentation | 1.0000 | 1.0000 | 170.02 seconds |
| With clitic segmentation | 1.0000 | 1.0000 | 164.74 seconds |

- CAMeL Tools D3 tokenization was used for Arabic clitic segmentation.
- The score did not increase because the original model already had a perfect score.
- The segmented model was slightly faster.

## Lab 5 — Search

| Configuration | recall@10 | MRR@10 | Average latency/query |
|---|---:|---:|---:|
| Bi-encoder only | 0.0590 | 0.0605 | 13.80 ms |
| + cross-encoder rerank | 0.0641 | 0.0590 | 58.49 ms |
| Arabic reranked slice | 0.0722 | 0.0530 | 65.39 ms |
| English reranked slice | 0.0571 | 0.0641 | 52.57 ms |

- No-answer empty-correct: 20 / 20
- Empty-result threshold: 0.4586
- Reranked recall cross-lingual gap: 0.0151
- Reranking improved recall slightly but reduced MRR.
- Reranking also increased the average latency.
- The search results did not reach the course targets.
- The measured results were kept without claiming an improvement.

## Lab 6 — Evaluation

| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| Topic classifier | Not run | Not run | Not run | Not run |
| Dialect-aware model | Not run | Not run | Not run | Not run |

- Step 1 completed: bootstrap confidence interval was implemented.
- Paired bootstrap difference was also implemented.
- All 6 bootstrap tests passed.
- Paired comparison verdict: Not run because only Step 1 was required.
- Error taxonomy top categories: Not completed.
- Top-3 prioritised fixes: Not completed.

## Lab 7 — Optimisation ladder

| Rung | p50 | p99 | Quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| PyTorch fixed padding @128 | 8.95 ms | 13.48 ms | Not measured | Not measured |
| PyTorch dynamic padding @128 | 6.62 ms | 11.44 ms | Not measured | Not measured |
| PyTorch dynamic padding @22 | 6.84 ms | 11.63 ms | Not measured | Not measured |
| ONNX fp32 @128 | Not run | Not run | Not run | Not run |
| ONNX INT8 @128 | Not run | Not run | Not run | Not run |

- Device: Tesla T4 GPU
- Number of benchmark examples: 500
- Recommended p95 length: 22 tokens
- Selected configuration: PyTorch dynamic padding with max length 128
- Dynamic padding reduced p50 latency by about 26%.
- Dynamic padding reduced p99 latency by about 15%.
- Reducing max length to 22 did not improve the speed in this run.
- HTTP p99 with 16 concurrent users: Not run
- Classifier quantisation decision: Not completed
- NER quantisation decision: Not completed