import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention --baseline version"""

    def __init__(self, dropout=0.0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q: [batch, head, 1, size]
        # k, v: [batch, head, seq_len, size]
        # mask: [batch, seq_len]

        attn = torch.matmul(q, k.transpose(-1, -2))  # [batch, head, 1, seq_len]

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)  # [batch, head, 1, seq_len]
        output = torch.matmul(attn, v)  # [batch, head, 1, size]

        return output, attn
