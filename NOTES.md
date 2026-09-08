# Lab Notes

## Lab 1 — Defect Safari
Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.

### Defect 1
- Class:
- Example:
- Why it matters:
- Decision:

### Defect 2
- Class:
- Example:
- Why it matters:
- Decision:

### Defect 3
- Class:
- Example:
- Why it matters:
- Decision:

### Defect 4
- Class:
- Example:
- Why it matters:
- Decision:

### Defect 5
- Class:
- Example:
- Why it matters:
- Decision:

### Defect 6
- Class:
- Example:
- Why it matters:
- Decision:

## Lab 2 — Parameter audit
| Checkpoint | Total params | Embeddings % | Other notes |
|---|---:|---:|---|
| mBERT | 177,853,440 | 51.84% | Large multilingual vocabulary makes embeddings the largest parameter bucket. |
| CAMeLBERT | 109,081,344 | 21.48% | Smaller Arabic-focused vocabulary reduces embedding parameters significantly. |

### Attention and masking findings

- The custom scaled dot-product attention matched the PyTorch reference at `atol=1e-6`.
- The causal attention matrix was lower triangular, confirming decoder-style causal attention.
- Future-token attention was zero after applying the causal mask.
- Average attention to the simulated padding token was `0.1729` before masking and `0.0` after masking.
- Padding masks are therefore required to prevent attention leakage into padding tokens.

## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:
