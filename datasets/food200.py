import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_FOOD200(path):
    df_train = list(
        csv.reader(open(
            path + '/ISIA_Food200_v2/metadata/train_finetune_v2.txt'
        ), delimiter=' ')
    )
    X_tr, Y_tr = zip(*df_train)
    Y_tr = list(map(int, Y_tr))

    df_val = list(
        csv.reader(open(
            path + '/ISIA_Food200_v2/metadata/val_finetune_v2.txt'
        ), delimiter=' ')
    )
    X_te, Y_te = zip(*df_val)
    Y_te = list(map(int, Y_te))

    return np.array(X_tr), torch.LongTensor(Y_tr), np.array(X_te), torch.LongTensor(Y_te)


class FOOD200_Labeled(Dataset):
    def __init__(self, X, Y, transform=None):
        self.root = 'data/ISIA_Food200_v2/imgs/'
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x = self.root + '/' + self.X[index]
        y = self.Y[index]

        x = Image.open(x).convert('RGB')
        x = self.transform(x)

        return x, y, index

    def __len__(self):
        return len(self.X)


class FOOD200_Unlabeled(Dataset):
    def __init__(self, X, transform=None):
        self.root = 'data/ISIA_Food200_v2/imgs/'
        self.X = X
        self.transform = transform

    def __getitem__(self, index):
        x = self.root + '/' + self.X[index]
        x = Image.open(x).convert('RGB')
        x = self.transform(x)
        return x, index

    def __len__(self):
        return len(self.X)
