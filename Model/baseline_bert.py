import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineBert(nn.Module):
    def __init__(self, input_size=768, linear_hidden_size=300, out_size=1, dropout=0.1):
        super(BaselineBert, self).__init__()

        # L-biLSTM
        self.lstm_hidden_size = 300
        self.num_layers = 2
        self.bert2emb = nn.Linear(input_size, self.lstm_hidden_size)
        self.bilstm = nn.LSTM(self.lstm_hidden_size, self.lstm_hidden_size, self.num_layers, bidirectional=True)  #
        # batch_first=False

        self.outlinear1 = nn.Linear(2 * self.lstm_hidden_size, linear_hidden_size)
        self.outlinear2 = nn.Linear(linear_hidden_size, out_size)
        self.dropout_out= nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        batch, seq_len, _ = x.shape
        x = self.bert2emb(x).transpose(0, 1)  # [seq_len, batch, input_size]
        # pack_padded_sequence?
        _, (h, _) = self.bilstm(x)
        h = h.view(self.num_layers, 2, batch, self.lstm_hidden_size)[-1]
        x = torch.cat((h[0], h[1]), dim=-1)  # [batch, 2 * lstm_hidden_size]

        x = self.outlinear1(x)
        x = F.relu(x)
        x= self.dropout_out(x)
        x = self.outlinear2(x)

        return x.squeeze()  # [batch]
