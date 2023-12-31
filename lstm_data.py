import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

PARITION_LENGTH = 140
MAX_LENGTH = PARITION_LENGTH + 1
END_TOKEN = np.array([0, 0, 0, 1])
PAD_TOKEN = np.array([0, 0, 0, 0])

class LSTMDataset(Dataset):
    def __init__(self, data, mean, std, parition=True) -> None:
        super().__init__()
        self.mean_x, self.mean_y = mean
        self.std_x, self.std_y = std

        if parition:
            data = self._parition(data)
        self.data = self._prepare(data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sequence = self.data[index]
        return sequence
    
    def revert(self):
        return self._revert(self.data)
    
    def _parition(self, data):
        paritioned = []
        for seq in data:
            length = len(seq)
            if length < PARITION_LENGTH:
                paritioned.append(seq)
            else:
                paritioned.append(seq[:length//2, :])
                paritioned.append(seq[length//2:, :])
        return paritioned
    
    def _prepare(self, data):
        size = len(data)
        prepared = np.zeros((size, MAX_LENGTH, 4))
        for i in range(size):
            seq_len = data[i].shape[0]
            prepared[i, :seq_len, 0] = (data[i][:, 0] + self.mean_x) / self.std_x
            prepared[i, :seq_len, 1] = (data[i][:, 1] + self.mean_y) / self.std_y
            prepared[i, :seq_len, 2] = data[i][:, 2]
            prepared[i, seq_len, :] = END_TOKEN
            prepared[i, seq_len+1:, :] = PAD_TOKEN
        prepared = torch.from_numpy(prepared).float()
        return prepared

    def _revert(self, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().numpy()
        reverted = []
        for i in range(len(data)):
            end_indices = (np.round(data[i, :, 3]) == 1).nonzero()[0]
            end_index = end_indices[0] if len(end_indices) > 0 else MAX_LENGTH
            reverted.append(data[i, :end_index, :3])
            reverted[i][:, 0] = (reverted[i][:, 0] * self.std_x) - self.mean_x
            reverted[i][:, 1] = (reverted[i][:, 1] * self.std_y) - self.mean_y
            reverted[i] = np.round(reverted[i]).astype(np.int16)
        reverted = np.array(reverted, dtype=object)
        return reverted
    
def revert(data, mean, std):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = np.copy(data)
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    reverted = []
    for i in range(len(data)):
        end_indices = (np.round(data[i, :, 3]) == 1).nonzero()[0]
        end_index = end_indices[0] if len(end_indices) > 0 else MAX_LENGTH
        reverted.append(data[i, :end_index, :3])
        reverted[i][:, 0] = (reverted[i][:, 0] * std[0]) - mean[0]
        reverted[i][:, 1] = (reverted[i][:, 1] * std[1]) - mean[1]
        reverted[i] = np.round(reverted[i]).astype(np.int16)
    reverted = np.array(reverted, dtype=object)
    if reverted.shape[0] == 1:
        reverted = np.squeeze(reverted, axis=0)
    return reverted

def create_mask(data):
    end_token_index = 3
    # Locate the end token by finding where the third component is 1
    end_token_positions = (data[:, end_token_index] == 1).nonzero(as_tuple=True)
    
    if end_token_positions[0].nelement() == 0:
        # If there is no end token, return a mask of all True
        end_index = MAX_LENGTH
    else:
        # Get the index of the first end token
        end_index = end_token_positions[0][0]
    # Create a mask that is True up to the end token and False afterwards
    mask = torch.arange(data.size(0)) < end_index
    mask = mask.reshape(-1, 1).to(data.device)
    return

def create_batch_mask(data):
    end_token_index = 3
    
    # Find the indices of the end token for each sequence in the batch
    end_token_indices = (data[..., end_token_index] == 1).long().argmax(dim=1)

    # Create a range tensor that broadcasts across the batch
    sequence_range = torch.arange(data.size(1)).unsqueeze(0).to(data.device)

    # Use broadcasting to compare the range with the indices and create the mask
    mask = sequence_range < end_token_indices.unsqueeze(1)
    mask = mask.reshape(data.size(0), -1, 1).to(data.device)
    return mask
