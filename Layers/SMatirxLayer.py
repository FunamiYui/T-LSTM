import torch.nn as nn
import torch.nn.functional as F
import torch
import torch
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data


# graph functional
class SentenceMatrixLayer(nn.Module):
    def __init__(self, in_size, out_size=1, p_Asem=0.6):
        super(SentenceMatrixLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.p_Asem = p_Asem
        self.linear = nn.Linear(in_size * 2, out_size)

    def forward(self, x, adj, mask):
        # x: [batch, seq_len, embed_dim]
        # adj: [batch, seq_len, seq_len], dense
        # mask: [batch, seq_len + 2]

        # adj is dense batch*node*node*(2*emb)
        # 2*emb for cat xi,xj
        # new_adj = adj.unsqueeze(-1)
        # new_adj = new_adj.expand(new_adj.shape[0], new_adj.shape[1], new_adj.shape[2], x.shape[-1] * 2)

        seq_len = x.shape[1]
        xi = x.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [batch, seq_len, 1, embed_dim]
        xj = x.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [batch, 1, seq_len, embed_dim]

        xij = torch.sigmoid(self.linear(torch.cat((xi, xj), dim=-1))).squeeze(-1)  # [batch, seq_len, seq_len]
        A_esm = self.p_Asem * xij + (1 - self.p_Asem) * adj

        mask = mask[:, 2:]  # [batch, seq_len]
        assert mask.shape[1] == seq_len, "seq_len inconsistent"
        mask_i = mask.unsqueeze(1).expand(-1, seq_len, -1).bool()  # [batch, 1, seq_len]
        mask_j = mask.unsqueeze(2).expand(-1, -1, seq_len).bool()  # [batch, seq_len, 1]
        A_mask = torch.bitwise_and(mask_i, mask_j)

        return A_esm.masked_fill(A_mask, 1e-9)  # [batch, seq_len, seq_len]

##test
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# x = torch.rand((3, 100))
# tri = torch.rand((1, 72))
# data = Data(x=x, edge_index=edge_index)
# device = torch.device('cuda')
# data = data.to(device)
# tri = tri.to(device)
# model = FRGN(100, 1)
# model.cuda()
# test = model(data)
# print(test)
