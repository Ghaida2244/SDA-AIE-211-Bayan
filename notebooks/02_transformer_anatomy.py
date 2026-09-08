"""Lab 2 starter notebook-as-script.
Complete the marked sections, verify numerical equivalence, inspect parameter
accounting, causal masking, attention heads and pad-attention leakage.
"""

import math

import torch
import torch.nn.functional as F
from bayan.attention import attention, MultiHeadAttention


def main():
    torch.manual_seed(42)

    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

   # ناتج الدالة التي بنيناها
    our_output = attention(q, k, v)

   # ناتج PyTorch المرجعي
    pytorch_output = F.scaled_dot_product_attention(q, k, v)

    # مقارنة النتيجتين
    matches = torch.allclose(
       our_output,
       pytorch_output,
       atol=1e-6,
    )

    # حساب درجات وأوزان Attention للفحص
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(q.size(-1))

    weights = torch.softmax(scores, dim=-1)

    print("\nAttention weights for the first head:")
    print(weights[0, 0])

    print("\nRow sums:")
    print(weights[0, 0].sum(dim=-1))
    # إنشاء Causal Mask سفلي مثلثي
    sequence_length = q.size(-2)

    causal_mask = torch.tril(
       torch.ones(
           sequence_length,
           sequence_length,
           dtype=torch.bool,
      )
    )

    # تشغيل Attention باستخدام القناع
    causal_output = attention(q, k, v, mask=causal_mask)

    # حساب الأوزان المقنّعة لعرضها
    masked_scores = scores.masked_fill(
      ~causal_mask,
      float("-inf"),
    )

    causal_weights = torch.softmax(masked_scores, dim=-1)

    print("\nCausal mask:")
    print(causal_mask)

    print("\nCausal attention weights for the first head:")
    print(causal_weights[0, 0])

    print("\nCausal output shape:")
    print(causal_output.shape)
    

    print("Matches PyTorch:", matches)
    print("Output shape:", our_output.shape)
    # التأكد أن الانتباه للمستقبل يساوي صفرًا
    future_weights = causal_weights.masked_select(~causal_mask)

    print(
       "\nFuture attention is zero:",
       torch.all(future_weights == 0).item(),
    )

    # تجربة Multi-Head Attention
    mha = MultiHeadAttention(
       embed_dim=8,
       num_heads=2,
    )

    mha_input = torch.randn(1, 4, 8)

    mha_output = mha(
       mha_input,
       mha_input,
       mha_input,
    )

    print("\nMulti-Head Attention output shape:")
    print(mha_output.shape)


    # الصقي كود Pad-Attention Leakage هنا
    pad_mask = torch.tensor(
        [True, True, True, False],
        dtype=torch.bool,
    ).view(1, 1, 1, 4)

    pad_attention_before = weights[..., -1].mean()

    pad_masked_scores = scores.masked_fill(
        ~pad_mask,
        float("-inf"),
    )

    pad_masked_weights = torch.softmax(
        pad_masked_scores,
        dim=-1,
    )

    pad_attention_after = pad_masked_weights[..., -1].mean()

    pad_masked_output = attention(
        q,
        k,
        v,
        mask=pad_mask,
    )

    print("\nAverage attention to PAD before masking:")
    print(pad_attention_before.item())

    print("\nAverage attention to PAD after masking:")
    print(pad_attention_after.item())

    print("\nPAD-masked output shape:")
    print(pad_masked_output.shape)


if __name__ == "__main__":
    main()
   





