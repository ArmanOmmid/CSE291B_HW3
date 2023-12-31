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

class LSTMGenerator0(_LSTM):
    def __init__(self, h_dim=64, dim=4, num_layers=4, sequential_channels=None, bidirectional=None, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(h_dim, h_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(h_dim, dim)

    def forward(self, z):
        inputs = z
        output, _ = self.lstm(inputs)
        output = self.fc(output)
        
        x_y = torch.tanh(output[:, :, :2]) * 3 
        b_p = torch.sigmoid(output[:, :, 2:])
        output = torch.cat((x_y, b_p), dim=2)

        return output

class LSTMDiscriminator0(_LSTM):
    def __init__(self, h_dim=64, dim=4, num_layers=4, sequential_channels=None, bidirectional=None,  **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(dim, h_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(h_dim, 1)

    def forward(self, inputs):
        # hidden, cell = self.init_hidden(inputs.size(0))

        output, (hidden, cell) = self.lstm(inputs)
        output = self.fc(hidden[-1]) # Last Layers hidden states

        return output
    