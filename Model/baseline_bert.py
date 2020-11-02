import torch.nn as nn
import torch.nn.functional as F

class BaselineBert(nn.Module):
    def __init__(self, bert_hidden_dim=768, linear_hidden_size=300, out_size=1, dropout=0.1):
        super(BaselineBert, self).__init__()
        self.outlinear1 = nn.Linear(bert_hidden_dim, linear_hidden_size)
        self.outlinear2 = nn.Linear(linear_hidden_size, out_size)
        self.dropout_out= nn.Dropout(dropout)

    def forward(self, x):
        x = self.outlinear1(x)
        x = F.relu(x)
        x= self.dropout_out(x)
        x = self.outlinear2(x)

        return x.squeeze()
