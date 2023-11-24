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

    def save(self, save_path):
        """
        All saves should be under the same path folder, under different tag folders, with the same filename
        """
        save_path = save_path.split(".")[0] + WEIGHTS_EXTENSION
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

class SequentialTanh(nn.Module):
    def __init__(self, channel_dims, logits=True, **kwargs):
        super().__init__(**kwargs)

        self.sequential = []
        prev_dim = channel_dims[0]
        module_count = len(channel_dims) - 1
        for i, dim in enumerate(channel_dims[1:]):
            self.sequential.append(nn.Linear(prev_dim, dim))
            if i < module_count-1 or not logits:
                self.sequential.append(nn.Tanh())
            prev_dim = dim
        self.sequential = nn.Sequential(*self.sequential)

    def forward(self, x):
        return self.sequential(x)

class LSTMGenerator(_LSTM):
    def __init__(self, h_dim=64, dim=4, num_layers=4, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(h_dim, h_dim, num_layers, batch_first=True)

        sequential_channels = [h_dim, h_dim//2, h_dim//4, h_dim//8, dim]
        self.sequential = SequentialTanh(sequential_channels)

    def forward(self, inputs):

        batch_size, seq_len = inputs.size(0), inputs.size(1)

        output, (hidden, cell) = self.lstm(inputs)

        output = output.contiguous().view(-1, self.h_dim)
        output = self.sequential(output)
        output = output.view(batch_size, seq_len, -1)
        
        x_y = torch.tanh(output[:, :, :2]) * 3 
        b_p = torch.sigmoid(output[:, :, 2:])
        output = torch.cat((x_y, b_p), dim=2)

        return output

class LSTMDiscriminator(_LSTM):
    def __init__(self, h_dim=64, dim=4, num_layers=4, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(dim, h_dim, num_layers, batch_first=True)

        sequential_channels = [h_dim, h_dim//2, h_dim//4, dim]
        self.sequential = SequentialTanh(sequential_channels)

    def forward(self, inputs):

        batch_size, seq_len = inputs.size(0), inputs.size(1)

        output, (hidden, cell) = self.lstm(inputs)
        
        output = output.contiguous().view(-1, self.h_dim)
        output = self.sequential(output)
        output = output.view(batch_size, seq_len, -1)

        return output
    