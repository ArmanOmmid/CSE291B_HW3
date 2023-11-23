from abc import ABC

import os
import torch
import torch.nn as nn

WEIGHTS_EXTENSION = ".pth"

class _Network(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, weights_path: str=None, map_location: str="cpu"):
        """
        Load weights if weights were specified
        """
        if not weights_path: return
        self.load_state_dict(torch.load(weights_path, map_location=torch.device(map_location)))

    def save(self, path: str, tag: str, filename: str="weights.pth"):
        """
        All saves should be under the same path folder, under different tag folders, with the same filename
        """
        filename = filename.split(".")[0] + WEIGHTS_EXTENSION
        save_path = os.path.join(path, tag, filename)
        torch.save(self.state_dict(), save_path)

class _LSTM(_Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_device(self):
        return next(self.parameters()).device
    
    def init_hidden(self, batch_size):
        device = self.get_device()
        h0 = torch.zeros(self.num_layers, batch_size, self.h_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.h_dim, device=device)
        return (h0, c0)

class LSTMGenerator(_LSTM):
    def __init__(self, h_dim=64, dim=4, num_layers=4, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(h_dim, h_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(h_dim, dim)

    def forward(self, z, seq_lengths=251):
        batch_size = z.size(0)
        hidden, cell = self.init_hidden(batch_size)

        outputs = torch.zeros(batch_size, seq_lengths, self.dim, device=self.get_device())

        inputs = z.unsqueeze(1) # Add L dim
        for i in range(seq_lengths):
            output, (hidden, cell) = self.lstm(inputs, (hidden, cell))
            inputs = output
            output = self.fc(output)
            outputs[:, i, :] = output.squeeze(1)

        return outputs

class LSTMDiscriminator(_LSTM):
    def __init__(self, h_dim=64, dim=4, num_layers=4, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(dim, h_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(h_dim, 1)

    def forward(self, inputs):
        hidden, cell = self.init_hidden(inputs.size(0))

        output, (hidden, cell) = self.lstm(inputs, (hidden, cell))
        output = self.fc(hidden[-1]) # Last Layers hidden states

        return output
    