import math
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np


# 0) prepare dataset
class WineDataset(Dataset):
    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt(fname="./data/wine/wine.csv", delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]  # (n_samples, 1) - shape of the array after putting the 0 in brackets
        self.n_samples = xy.shape[0]
        self.n_features = xy.shape[1]

        self.transform = transform

    def __getitem__(self, index):
        # retrieve item; indexing
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets


dataset = WineDataset(transform=ToTensor())
# dataset = WineDataset(transform=None)
# dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)
# dataiter = iter(dataloader)
# data = next(dataiter)
features, label = dataset[0]
print(features, label, end="\n\n")

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
new_dataset = WineDataset(transform=composed)
features, label = new_dataset[0]
print(features, label)

exit(1)
# 1) prepare model

# 2) loss and optimizer

# 3) training loop
num_epochs = 10
total_samples = len(dataset)
n_iters = math.ceil(total_samples / dataloader.batch_size)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward and backward pass, update
        if not (i + 1) % 5:
            print(f"Epoch {epoch + 1}/{num_epochs}: Step {i + 1}/{n_iters}, inputs {inputs.shape}")

# torchvision.datasets.MNIST()
# torchvision.datasets.FashionMNIST()
