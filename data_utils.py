import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CryptoDataset(Dataset):
    def __init__(self, X_data, y_data, sequence_len):
        self.sequence_len = sequence_len
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        X = torch.FloatTensor(self.X_data[index : index + self.sequence_len])
        """ The target shape is [bs, 1] and not [bs, seq_len] because the LSTM model is being trained to 
            predict a single value, i.e., the next closing price of the cryptocurrency, given a sequence 
            of historical prices. """
        y = torch.FloatTensor(np.array(self.y_data[index + self.sequence_len]))
        
        return X, y

    def __len__(self):
        return len(self.X_data) - self.sequence_len





