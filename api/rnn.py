from typing import Sequence
from numpy.core import numeric
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.val_linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)

        out, _ = self.rnn(x, h0)

        out = out[:, -1, :]

        out = self.val_linear(out)

        out = torch.cat(out, dim=1)

        return out
