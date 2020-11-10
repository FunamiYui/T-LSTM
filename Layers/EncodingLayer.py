import torch
import torch.nn as nn
import torch.nn.functional as F


# graph functional
class EncodeLayer(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=2, bi=True):
        super(EncodeLayer, self).__init__()
        self.bilstm = nn.LSTM(in_size, hidden_size, num_layers=num_layers, bidirectional=bi)  # batch_first=False

    def forward(self, x, mask):
        # x: [batch, seq_len, bert_dim]
        # mask: [batch, seq_len]
        seq_len = x.shape[1]
        x = x.transpose(0, 1)  # [seq_len, batch, input_size]

        seq_lens = torch.sum(mask, dim=-1, dtype=torch.long)
        sorted_seq_lens, indices = torch.sort(seq_lens, descending=True)
        _, desorted_indices = torch.sort(indices)
        x = x[:, indices]
        x = nn.utils.rnn.pack_padded_sequence(x, sorted_seq_lens)

        out, _ = self.bilstm(x)

        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        out = out[:, desorted_indices]

        out = out.transpose(0, 1)  # [batch, seq_len, 2 * lstm_hidden_size]
        batch, new_seq_len, output_size = out.shape
        if new_seq_len < seq_len:
            out = torch.cat((out, torch.zeros([batch, seq_len - new_seq_len, output_size]).cuda()), dim=1)
        return out  # [batch, seq_len, 2 * lstm_hidden_size]


