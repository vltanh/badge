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


class CIFAR10_Labeled(Dataset):
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


class CIFAR10_Unlabeled(Dataset):
    def __init__(self, X, transform=None):
        self.X = X
        self.transform = transform

    def __getitem__(self, index):
        x = self.X[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, index

    def __len__(self):
        return len(self.X)


class CIFAR10_Adversarial(Dataset):
    def __init__(self, X_1, Y_1, X_2, Y_2, transform=None):
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):
        return max(len(self.X1), len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]
        else:
            re_index = index % Len1
            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        if index < Len2:
            x_2 = self.X2[index]
            y_2 = self.Y2[index]
        else:
            re_index = index % Len2
            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        if self.transform is not None:
            x_1 = Image.fromarray(x_1)
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(x_2)
            x_2 = self.transform(x_2)

        return index, x_1, y_1, x_2, y_2
