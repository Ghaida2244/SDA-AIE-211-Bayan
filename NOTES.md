# Lab Notes

## Lab 1 — Defect Safari

I checked the raw Bayan data before writing the preprocessing code. I found the following text problems.

### Defect 1 — Extra Spaces

- Example: `  ألعاب الأطفال في حديقة حي العليا تحتاج صيانة   `
- Problem: Extra spaces make the text inconsistent.
- Decision: Remove spaces from the beginning and end and change repeated spaces into one space.

### Defect 2 — Repeated Characters

- Example: `لووووسمحت`
- Problem: Repeated characters can create too many tokens.
- Decision: Keep a maximum of two repeated characters because repetition may show emphasis.

### Defect 3 — Personal Information

- Examples: `0551234567` and `1023456789`
- Problem: Phone numbers and national ID values should not be sent to the models.
- Decision: Replace phone numbers with `<PHONE>` and national IDs with `<NATIONAL_ID>`.

### Defect 4 — HTML Text

- Example: `<br>`
- Problem: HTML tags are not part of the real complaint.
- Decision: This should be cleaned from the original data source. HTML removal was not required by the Lab 1 tests.

### Defect 5 — Emoji

- Example: `الخدمة سيئة 😡`
- Problem: Removing emoji may remove useful sentiment information.
- Decision: Keep emoji because they may show the user's opinion or emotion.

### Defect 6 — Arabic and English in the Same Text

- Example: Arabic text that contains a reference such as `BYN-2026-000035`.
- Problem: Changing or removing English text may remove useful service names or reference numbers.
- Decision: Keep Arabic and English text and preserve English capitalization.

### Preprocessing Results

The preprocessing code performs the following steps:

- Apply NFC Unicode normalization.
- Remove Arabic tatweel.
- Reduce character repetition to a maximum of two.
- Remove extra whitespace.
- Preserve emoji.
- Preserve English uppercase and lowercase letters.
- Mask phone numbers.
- Mask Saudi national-ID-shaped values.

Test results:

- Golden preprocessing: `25/25 passed`
- PII masking: `60/60 = 100%`

### Sentence Segmentation

I used a blank multilingual spaCy pipeline with the `sentencizer`.

The pipeline:

- Applies preprocessing first.
- Splits normal Arabic sentences correctly.
- Splits normal English sentences correctly.
- Masks phone numbers before splitting.
- Removes tatweel.
- Preserves emoji.

One limitation was found with numbered lists. For example:

`1. فتحت التطبيق. 2. ظهر خطأ.`

The sentencizer separates `1.` and `2.` from their sentences. A custom rule may be needed later if numbered complaints are common.

### Tokenizer Audit

I compared four tokenizers:

- mBERT
- XLM-R
- CAMeLBERT
- DistilBERT

The results were:

| Tokenizer | AR fertility | EN fertility | AR p95 | EN p95 | AR UNK rate |
|---|---:|---:|---:|---:|---:|
| mBERT | 2.15 | 1.51 | 27 | 25 | 0.005 |
| XLM-R | 1.67 | 1.43 | 21 | 23 | 0.000 |
| CAMeLBERT | 1.41 | 2.70 | 20 | 38 | 0.008 |
| DistilBERT | 4.53 | 1.30 | 47 | 21 | 0.002 |

I selected XLM-R because it gave balanced results for both Arabic and English.

CAMeLBERT was better for Arabic, but it divided English text into more pieces. DistilBERT was good for English, but it was not good for Arabic.

---

## Lab 2 — Transformer Anatomy

### Scaled Dot-Product Attention

I implemented Scaled Dot-Product Attention using the following steps:

1. Multiply Query by the transpose of Key.
2. Divide the scores by the square root of the key dimension.
3. Apply the mask when it is available.
4. Apply Softmax.
5. Multiply the attention weights by Value.

The equation is:

`Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V`

The result of my implementation matched the PyTorch result with a tolerance of `1e-6`.

The sum of every row in the attention-weight matrix was approximately `1.0`.

### Multi-Head Attention

I implemented Multi-Head Attention using:

- Query projection
- Key projection
- Value projection
- Multiple attention heads
- Final output projection

Example result:

- Input shape: `torch.Size([1, 4, 8])`
- Output shape: `torch.Size([1, 4, 8])`

This showed that the heads were divided and combined correctly.

### Causal Mask

I created a lower-triangular causal mask.

The mask allows token position `i` to attend only to the current and previous positions.

The attention values for future positions became zero.

This type of attention is used in decoder-style models.

### Padding Attention

I tested the attention given to a simulated padding token.

Results:

- Average attention to padding before masking: `0.1729`
- Average attention to padding after masking: `0.0`

This showed that the model can attend to padding when the correct mask is missing.

The padding mask prevents the model from using padding tokens.

### Attention Map Note

The attention example used random Query, Key, and Value tensors.

It was useful for checking:

- Attention calculations
- Causal masking
- Padding masking
- Multi-head output shapes

The example did not contain real tokenizer tokens such as `[SEP]`. Therefore, `[SEP]` attention behavior was not checked in this experiment.

### Parameter Audit

I compared the parameter counts of mBERT and CAMeLBERT.

| Checkpoint | Total parameters | Embedding parameters | Embedding share |
|---|---:|---:|---:|
| mBERT | 177,853,440 | 92,206,848 | 51.84% |
| CAMeLBERT | 109,081,344 | 23,434,752 | 21.48% |

Other parameter groups:

| Parameter group | mBERT | CAMeLBERT |
|---|---:|---:|
| Attention | 28,348,416 | 28,348,416 |
| FFN | 56,669,184 | 56,669,184 |
| Normalization | 38,400 | 38,400 |
| Pooler | 590,592 | 590,592 |

The main difference is in the embedding layer.

mBERT supports many languages and has a larger vocabulary. Because of this, its embedding layer uses more parameters. CAMeLBERT focuses on Arabic and has a smaller vocabulary.

---

## Lab 3 — Model Training and Evaluation

### Grouped Dataset Split

I divided the topic data into:

- Training set
- Validation set
- Test set

I checked the `citizen_group_id` values in each set.

There was no citizen overlap between the three sets.

This is important because the same citizen should not appear in both training and evaluation data. Otherwise, the result may be higher than the real model performance.

The grouped-split test passed:

- `1/1 passed`

### TF-IDF Baseline

I created a simple baseline using:

- TF-IDF
- LinearSVC
- Maximum vocabulary size of 20 features

Results:

- Validation macro-F1: `0.7106`
- Frozen-test macro-F1: `0.4699`
- Vocabulary size: `20`
- Training time: `0.1455 seconds`

This baseline was used to check whether the Transformer model gave a useful improvement.

### Topic Classification Model

I used:

- Checkpoint: `xlm-roberta-base`
- Maximum sequence length: `128`
- Epochs: `3`
- Learning rate: `2e-5`
- Training batch size: `16`
- Validation batch size: `32`

Dataset sizes:

- Training examples: `8400`
- Validation examples: `2400`
- Test examples: `1200`

Results:

- Validation macro-F1: `1.0000`
- Frozen-test macro-F1: `1.0000`
- Training time: `276.79 seconds`

Improvement over the baseline:

- Validation improvement: `1.0000 - 0.7106 = +0.2894`
- Frozen-test improvement: `1.0000 - 0.4699 = +0.5301`

The required improvement was at least `+0.08`, so the model passed the target.

The trained model was saved in Google Drive.

### NER Label Alignment

The NER labels were originally connected to complete words.

XLM-R may split one word into more than one subword. I used `align_labels()` to connect the labels to the correct subwords.

The rules were:

- Give the original label to the first subword.
- Give `-100` to additional subwords.
- Give `-100` to special tokens.
- Give `-100` to padding tokens.

The value `-100` tells the loss function to ignore that position.

Results:

- NER alignment tests: `8/8 passed`

I also found that some CoNLL tokens contained spaces, such as:

`خدمات المياه`

For this reason, I split each CoNLL line once from the right. This kept the token text together and separated only the NER label.

### NER Model

I used:

- Checkpoint: `xlm-roberta-base`
- Maximum sequence length: `128`
- Epochs: `3`
- Learning rate: `2e-5`
- Training batch size: `16`
- Validation batch size: `32`

The NER labels were:

- `B-DATE`
- `B-LOCATION`
- `B-REFERENCE`
- `B-SERVICE`
- `O`

Dataset sizes:

- Total sentences: `4000`
- Training sentences: `3200`
- Validation sentences: `400`
- Test sentences: `400`

Results:

- Validation entity-level F1: `1.0000`
- Frozen-test entity-level F1: `1.0000`
- Training time: `170.02 seconds`

The required entity-level F1 was at least `0.80`, so the model passed the target.

The trained NER model was saved in Google Drive.

### Extractive QA

I implemented `best_span()` to select the best answer from the context.

The function checks that:

- The start and end positions are valid.
- The end position does not appear before the start.
- The answer is not longer than the maximum allowed length.
- Special tokens are not selected as answers.
- Empty spans are rejected.
- The model can return no answer when the null score is stronger.

QA unit-test results:

- `2/2 passed`

### QA Smoke Test

I used:

- Checkpoint: `deepset/xlm-roberta-base-squad2`
- Answerable questions: `9`
- Unanswerable questions: `3`

The supplied smoke file contained answerable questions only. I selected three unanswerable questions from `data/models/bayan_qa.json` to complete the required test.

Some correct model answers included extra spaces or the English article `the`.

For example:

- Predicted: `the Bayan portal`
- Expected: `Bayan portal`

I used SQuAD-style answer normalization before comparison. It removes:

- English capitalization differences
- Punctuation
- Extra spaces
- English articles such as `a`, `an`, and `the`

Final results:

- Answerable questions: `9/9 correct`
- Unanswerable questions: `3/3 returned None`
- QA smoke test: `passed`

### Lab 3 Contract Tests

The final local tests were:

- Grouped dataset split: passed
- NER label alignment: passed
- QA span selection: passed
- Total result: `11 passed`

### Training Environment

The local code and tests used Python `3.12`.

The Transformer models were trained using:

- Google Colab
- Tesla T4 GPU

The Colab environment used Python `3.13`, while the project doctor expected Python `3.12`.

For this reason, the final contract tests were checked locally using Python `3.12`.

Large trained model files were saved in Google Drive instead of GitHub.

---

## Lab 4 — Dialect Audit

- Distribution:
- One-sentence implication for MSA-only evaluation: