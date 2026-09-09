# Decision Records

## tokenizer
- Chosen checkpoint(s): `xlm-roberta-base` (XLM-R)
- Arabic fertility evidence:XLM-R achieved 1.67 pieces per word. It performed better than mBERT at 2.15 and DistilBERT at 4.53, although CAMeLBERT achieved the lowest Arabic fertility at 1.41. 
- English fertility evidence: XLM-R achieved 1.43 pieces per word, providing better English tokenisation than CAMeLBERT at 2.70 and remaining close to DistilBERT at 1.30.
- p95 length evidence: XLM-R achieved an Arabic p95 length of 21 tokens and an English p95 length of 23 tokens, giving balanced sequence lengths across both languages.
- Operational trade-off / rationale: XLM-R provides the best bilingual balance for Bayan. It has low fertility in both Arabic and English, balanced sequence lengths, and an Arabic unknown-token rate of 0.000. CAMeLBERT performs slightly better for Arabic but fragments English text more heavily, while DistilBERT performs poorly on Arabic.

## arabic-model
- Incumbent: CAMeLBERT-DA
- Candidate: CAMeLBERT-mix
- All/Gulf/MSA evidence: Both models achieved 1.0000 macro-F1 on the full Arabic test set, the Gulf slice, and the MSA slice. CAMeLBERT-mix trained in 102.60 seconds, compared with 121.19 seconds for CAMeLBERT-DA.
- CI-backed verdict: No accuracy improvement could be established because both models reached the maximum score. CAMeLBERT-mix is selected based on equal slice performance and lower training time. This result is affected by a ceiling effect in the synthetic dataset.
- Segmentation contract: Arabic clitics are segmented with CAMeL Tools using the D3 tokenization scheme. Segmentation must be applied consistently during training, evaluation, and serving. The segmented NER model achieved 1.0000 validation and frozen-test entity F1.

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


## Search model
- Bi-encoder: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
- Decision: Keep the bi-encoder as the default retrieval stage. Reranking is optional because it slightly reduced MRR and added about 44.69 ms of average latency.
- Empty-result threshold: 0.4586
- Limitation: The measured retrieval quality did not reach the course targets.