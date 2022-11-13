import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt(fname="./data/wine/wine.csv", delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]     # (n_samples, 1) - shape of the array after putting the 0 in brackets
        self.n_samples = xy.shape[0]
        self.n_features = xy.shape[1]

    def __getitem__(self, index):
        # retrieve item; indexing
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


dataset = WineDataset()
first = 