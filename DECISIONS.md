# Decision Records

## tokenizer
- Chosen checkpoint(s): `xlm-roberta-base` (XLM-R)
- Arabic fertility evidence:XLM-R achieved 1.67 pieces per word. It performed better than mBERT at 2.15 and DistilBERT at 4.53, although CAMeLBERT achieved the lowest Arabic fertility at 1.41. 
- English fertility evidence: XLM-R achieved 1.43 pieces per word, providing better English tokenisation than CAMeLBERT at 2.70 and remaining close to DistilBERT at 1.30.
- p95 length evidence: XLM-R achieved an Arabic p95 length of 21 tokens and an English p95 length of 23 tokens, giving balanced sequence lengths across both languages.
- Operational trade-off / rationale: XLM-R provides the best bilingual balance for Bayan. It has low fertility in both Arabic and English, balanced sequence lengths, and an Arabic unknown-token rate of 0.000. CAMeLBERT performs slightly better for Arabic but fragments English text more heavily, while DistilBERT performs poorly on Arabic.

## arabic-model
- Incumbent:
- Candidate:
- All/Gulf/MSA evidence:
- CI-backed verdict:
- Segmentation contract:

## search-min-score
- Threshold:
- No-answer evidence:
- False-positive / false-negative trade-off:

## quantisation-split
- Topic artefact:
- NER artefact:
- Latency evidence:
- Paired quality-tax evidence:
- Rollback artefact retained:

## architecture
- Encoder/decoder rationale by task:
- Multilingual vs Arabic-centric rationale:
- Evidence used:
