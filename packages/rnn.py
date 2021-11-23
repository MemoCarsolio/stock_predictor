# PyTorch imports
import torch
import torch.nn as nn

device = torch.device('cpu')


class RNN(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            num_classes
    ):
        """
        RNN Module
        -------------------

        Parameters
        ----------
        input_size: int
        hidden_size: int
        num_layers: int
        num_clases: int


        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.val_linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)

        out, _ = self.gru(x, h0)

        out = out[:, -1, :]

        out = self.val_linear(out)

        return out
