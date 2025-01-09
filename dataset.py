import torch
from torch.utils.data import Dataset

class PathologyDataset(Dataset):
    def __init__(self, logs, packets, labels):
        self.logs = logs
        self.packets = packets
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.logs[idx], self.packets[idx], self.labels[idx]
