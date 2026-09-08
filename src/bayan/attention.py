"""Lab 2 starter: scaled dot-product attention and multi-head attention."""

import math
import torch



def attention(q, k, v, mask=None):
    # حساب التشابه بين Query وKey
    scores = torch.matmul(q, k.transpose(-2, -1))

    # تصغير القيم لمنع Softmax من أن تصبح شديدة جدًا
    d_k = q.size(-1)
    scores = scores / math.sqrt(d_k)

    # تطبيق القناع عند توفيره
    if mask is not None:
      if mask.dtype == torch.bool:
           scores = scores.masked_fill(~mask, float("-inf"))
      else:
           scores = scores + mask

   # تحويل الدرجات إلى احتمالات
    weights = torch.softmax(scores, dim=-1)

    # استخدام الأوزان لاختيار المعلومات من Value
    output = torch.matmul(weights, v)

    return output


class MultiHeadAttention(torch.nn.Module):
   def __init__(self, embed_dim: int, num_heads: int):
    super().__init__()

    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")

    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads

    # طبقات تحويل المدخل إلى Query وKey وValue
    self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
    self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
    self.v_proj = torch.nn.Linear(embed_dim, embed_dim)

    # دمج نتائج جميع الرؤوس
    self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

   def split_heads(self, x):
    batch_size, seq_length, embed_dim = x.shape

    x = x.view(
        batch_size,
        seq_length,
        self.num_heads,
        self.head_dim,
    )

    return x.transpose(1, 2)

   def forward(self, q, k, v, mask=None):
    batch_size = q.size(0)

    # إنشاء Query وKey وValue
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)

    # تقسيم كل Tensor إلى عدة رؤوس
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)

    # تطبيق Attention على جميع الرؤوس
    attended = attention(q, k, v, mask)

    # إعادة دمج الرؤوس
    attended = attended.transpose(1, 2).contiguous()

    attended = attended.view(
        batch_size,
        -1,
        self.embed_dim,
    )

    # الإسقاط النهائي
    return self.out_proj(attended)