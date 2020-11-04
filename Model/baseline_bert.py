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

    def forward(self, x, trigger_index, mask=None):
        # x: [batch, seq_len, input_size]
        batch, seq_len, _ = x.shape
        x = self.bert2emb(x).transpose(0, 1)  # [seq_len, batch, input_size]
        # pack_padded_sequence?
        out, (_, _) = self.bilstm(x)
        out = out.transpose(0, 1)  # [batch, seq_len, 2 * lstm_hidden_size]

        diag_matrix = torch.diag(torch.ones(seq_len)).cuda()
        trigger_index = diag_matrix[trigger_index].bool().unsqueeze(-1).expand(-1, -1, out.shape[-1])
        x = out.masked_select(trigger_index).view(batch, -1)  # [batch, 2 * lstm_hidden_size]

        x = self.outlinear1(x)
        x = F.relu(x)
        x= self.dropout_out(x)
        x = self.outlinear2(x)

        return x.squeeze()  # [batch]
