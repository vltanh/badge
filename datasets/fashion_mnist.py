from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image


def get_FashionMNIST(path):
    raw_tr = datasets.FashionMNIST(path + '/FashionMNIST',
                                   train=True, download=True)
    raw_te = datasets.FashionMNIST(path + '/FashionMNIST',
                                   train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te


class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)
