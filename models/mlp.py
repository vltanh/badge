import torch
from torch import nn
from torch.nn import functional as F


class VGG_10_clf(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(512, 50)
        # self.fc2 = nn.Linear(50, 10)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        # e1 = F.relu(self.fc1(x))
        # x = F.dropout(e1, training=self.training)
        # x = self.fc2(x)
        # return x, e1
        return self.fc(x), None

    def get_embedding_dim(self):
        return 50


class VGG_10_dis(nn.Module):
    def __init__(self):
        super(VGG_10_dis, self).__init__()
        self.fc1 = nn.Linear(512, 50)
        self.fc2 = nn.Linear(50, 1)
        # self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
