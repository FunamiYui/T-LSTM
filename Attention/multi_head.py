import torch
import torch.nn as nn
import torch.nn.functional as F
from single import ScaledDotProductAttention


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_in, d_out, dropout=0.3):
        super().__init__()
        assert d_out % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_out // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(3)])
        self.output_linear = nn.Linear(d_out, d_out)
        self.attention = ScaledDotProductAttention(dropout)

        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        batch = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        # q, k, v: [batch, head, seq_len, embed_dim]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask)  # for head axis

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k)

        return self.output_linear(x)


if __name__ == "__main__":
    batch = 2
    seq_len = 3
    at_size = 10
    model = MultiHeadedAttention(1, at_size, at_size, 0.1)

    x = torch.randn([batch, seq_len, at_size])
    trigger_x = x[:, 1, :]
    # mask = torch.ones([batch, seq_len])
    x = model(trigger_x, x, x)
