# Bayan | بيان
## SDA-AIE-211 — Natural Language Processing with Transformers

Bayan is a bilingual Natural Language Processing project for analyzing citizen feedback in Arabic and English.

The project was developed as part of the **SDA-AIE-211: Natural Language Processing with Transformers** course at the **Saudi Digital Academy**.

Bayan includes text preprocessing, privacy protection, topic classification, named entity recognition, question answering, Arabic language analysis, semantic search, statistical evaluation, and inference benchmarking.

---

## Project Information

- Project name: Bayan | بيان
- Course: SDA-AIE-211
- Course topic: Natural Language Processing with Transformers
- Academy: Saudi Digital Academy
- Repository owner: Ghaida2244
- Main language: Python
- Recommended Python version: Python 3.12

---

## Project Goal

The goal of Bayan is to process Arabic and English citizen feedback and support several NLP tasks.

The system can:

- Clean and normalize bilingual text.
- Mask phone numbers and Saudi national-ID-shaped values.
- Preserve useful information such as emoji and punctuation.
- Split Arabic and English text into sentences.
- Classify feedback into service topics.
- Extract entities from Arabic text.
- Answer questions using extractive question answering.
- Handle questions that do not have an answer.
- Search for similar historical cases.
- Compare Arabic language models.
- Measure model quality and inference speed.

---

## Main Features

### Bilingual preprocessing

The preprocessing pipeline applies the same rules during training, evaluation, and inference.

It includes:

- Unicode normalization.
- Arabic and Persian digit conversion.
- Arabic diacritic removal.
- Tatweel removal.
- Repeated-character reduction.
- Whitespace cleanup.
- Phone number masking.
- National-ID-shaped value masking.
- Sentence segmentation.
- Preservation of emoji and useful punctuation.

The preprocessing version is:

```text
1.2.0
```

### Privacy protection

Supported private information is replaced before model use:

```text
<PHONE>
<NATIONAL_ID>
```

The PII test set contained 60 examples, and all examples were handled correctly.

### Topic classification

The project includes two topic classification approaches:

- TF-IDF with LinearSVC as a simple baseline.
- XLM-R as the Transformer classifier.

The classifier predicts one of eight topics:

```text
billing
digital_services
licensing
lighting
parks
roads
waste
water
```

### Named entity recognition

The NER model uses XLM-R to extract entities such as:

- Date
- Location
- Reference number
- Service

Subword labels are aligned correctly, while special tokens and continuation pieces use the ignored label value `-100`.

Arabic clitic segmentation is also supported with CAMeL Tools.

### Question answering

The question-answering component uses constrained span selection.

It checks that:

- The answer start is before the answer end.
- The answer does not exceed the maximum length.
- Invalid and special-token offsets are ignored.
- A null answer can be returned when the context does not contain an answer.

### Arabic language support

Two Arabic normalization profiles are available:

- A display profile that keeps the text readable.
- A Bayan Arabic profile for model input.

The Arabic dataset contains:

| Dialect | Examples | Percentage |
|---|---:|---:|
| Gulf | 4,800 | 66.67% |
| MSA | 2,400 | 33.33% |

CAMeLBERT-mix and CAMeLBERT-DA were compared on Arabic topic classification.

### Semantic search

Bayan includes bilingual semantic search using:

- A multilingual Sentence Transformer.
- L2-normalized embeddings.
- A FAISS cosine-similarity index.
- Persisted case metadata.
- A versioned index manifest.
- Optional cross-encoder reranking.
- A threshold for returning an honest empty result.

The index manifest records:

```text
model
preproc_version
n_vectors
dim
metric
normalized
```

### Statistical evaluation

Bootstrap confidence intervals and paired bootstrap differences are implemented.

Paired resampling keeps the two model results aligned during comparison.

### Inference benchmarking

The classifier was tested with:

- Fixed padding.
- Dynamic padding.
- Different maximum sequence lengths.
- Warm-up runs.
- Production-mix examples.
- p50 and p99 latency measurements.

---

## Models and Tools

| Task | Model or tool |
|---|---|
| Bilingual tokenizer and classifier | XLM-R |
| TF-IDF baseline | TF-IDF + LinearSVC |
| Named entity recognition | XLM-R |
| Question answering | XLM-RoBERTa SQuAD2 |
| Arabic model comparison | CAMeLBERT-mix and CAMeLBERT-DA |
| Arabic clitic segmentation | CAMeL Tools |
| Semantic search | Multilingual MiniLM |
| Search index | FAISS |
| Search reranking | Multilingual cross-encoder |
| Evaluation | NumPy and scikit-learn |
| Training | PyTorch and Hugging Face Transformers |

---

## Main Results

### Tokenizer audit

| Tokenizer | Arabic fertility | English fertility | Arabic p95 | English p95 | Arabic UNK rate |
|---|---:|---:|---:|---:|---:|
| mBERT | 2.15 | 1.51 | 27 | 25 | 0.005 |
| XLM-R | 1.67 | 1.43 | 21 | 23 | 0.000 |
| CAMeLBERT | 1.41 | 2.70 | 20 | 38 | 0.008 |
| DistilBERT | 4.53 | 1.30 | 47 | 21 | 0.002 |

XLM-R was selected because it provided balanced Arabic and English tokenization and had no measured Arabic unknown-token rate.

### Model results

| Model | Validation result | Test result |
|---|---:|---:|
| TF-IDF + LinearSVC | 0.7106 macro-F1 | 0.4699 macro-F1 |
| XLM-R topic classifier | 1.0000 macro-F1 | 1.0000 macro-F1 |
| XLM-R NER | 1.0000 entity-F1 | 1.0000 entity-F1 |
| QA smoke test | 12/12 correct | 12/12 correct |

### Arabic model comparison

| Model | All macro-F1 | Gulf macro-F1 | MSA macro-F1 | Training time |
|---|---:|---:|---:|---:|
| CAMeLBERT-mix | 1.0000 | 1.0000 | 1.0000 | 102.60 seconds |
| CAMeLBERT-DA | 1.0000 | 1.0000 | 1.0000 | 121.19 seconds |

CAMeLBERT-mix was selected because it achieved the same measured result in less training time.

### Search results

| Configuration | Recall@10 | MRR@10 | Average latency |
|---|---:|---:|---:|
| Bi-encoder | 0.0590 | 0.0605 | 13.80 ms |
| Cross-encoder reranking | 0.0641 | 0.0590 | 58.49 ms |

The empty-result threshold correctly rejected all 20 no-answer queries while retaining all 130 answerable queries.

The search quality did not reach the course target. These measured results were kept honestly and documented as a project limitation.

### Inference results

The classifier was measured on 500 production-mix examples using a Tesla T4 GPU.

| Configuration | p50 | p99 | Mean |
|---|---:|---:|---:|
| Fixed padding at 128 | 8.95 ms | 13.48 ms | 9.47 ms |
| Dynamic padding at 128 | 6.62 ms | 11.44 ms | 6.79 ms |
| Dynamic padding at 22 | 6.84 ms | 11.63 ms | 7.15 ms |

Dynamic padding at 128 was selected because it achieved the best measured latency.

---

## Repository Structure

```text
SDA-AIE-211-Bayan/
│
├── data/
│   ├── raw/
│   ├── eval/
│   ├── models/
│   ├── search/
│   └── serving/
│
├── notebooks/
│   ├── 00_colab_setup.ipynb
│   ├── 01_tokenizer_audit.py
│   ├── 02_transformer_anatomy.py
│   └── 05_retrieval_eval.py
│
├── scripts/
│   ├── arabic_bakeoff.py
│   ├── benchmark_inference.py
│   ├── dialect_audit.py
│   ├── parameter_audit.py
│   ├── qa_smoke.py
│   ├── tfidf_baseline.py
│   ├── train_classifier.py
│   └── train_ner.py
│
├── src/bayan/
│   ├── evaluation/
│   ├── models/
│   ├── preprocessing/
│   ├── search/
│   └── serving/
│
├── tests/
├── BENCHMARKS.md
├── DECISIONS.md
├── NOTES.md
├── requirements.txt
└── pyproject.toml
```

---

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```cmd
.venv\Scripts\activate
```

Install the requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Check the environment:

```bash
python scripts/doctor.py
```

---

## Running the Tests

Run the tests for the completed project components:

```bash
pytest tests/test_preprocessing.py tests/test_pii_recall.py tests/test_attention.py tests/test_model_data.py tests/test_ner_alignment.py tests/test_qa.py tests/test_arabic_normalize.py tests/test_search_contract.py tests/test_evaluation.py -q
```

---

## Running Main Components

Run the TF-IDF baseline:

```bash
python scripts/tfidf_baseline.py
```

Run the tokenizer audit:

```bash
python notebooks/01_tokenizer_audit.py
```

Build a search index:

```bash
python -c "from bayan.search.index import build_index; print(build_index(prefix='artifacts/case_index_v1'))"
```

Run retrieval evaluation:

```bash
python notebooks/05_retrieval_eval.py
```

Run the inference benchmark:

```bash
python scripts/benchmark_inference.py --model-dir PATH_TO_MODEL --samples 500
```

---

## Limitations

- Most of the project data is synthetic.
- Some models reached perfect results because the classification tasks were simple.
- The search component did not reach the target retrieval metrics.
- The search dataset contains repeated cases and incomplete relevance labels.
- Confidence intervals were implemented, but the complete evaluation report was not produced.
- ONNX export, INT8 quantization, HTTP load testing, and final API integration were outside the completed scope.
- Model artifacts are not stored in Git because of their large size.

---

## Responsible Use

Bayan masks supported private information before model processing.

The current PII rules support:

- Saudi mobile phone patterns.
- Saudi national-ID-shaped values.

The system should not be treated as a complete privacy solution without further testing and review.

---

## Course Attribution

This is an academic project created for:

**Saudi Digital Academy**  
**SDA-AIE-211 — Natural Language Processing with Transformers**

The project results shown in this repository are based on the student's own training, testing, and benchmarking runs.

---

## License

This project uses the license included in `LICENSE.txt`.