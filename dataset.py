import torch
from torch.utils.data import Dataset

class PathologyDataset(Dataset):
    def __init__(self, network_data, labels):
        self.network_data = network_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.network_data[idx], self.labels[idx]
