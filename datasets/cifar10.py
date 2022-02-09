import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image


def get_CIFAR10(path):
    data_tr = datasets.CIFAR10(path + '/CIFAR10',
                               train=True, download=True)
    data_te = datasets.CIFAR10(path + '/CIFAR10',
                               train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te


class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)
