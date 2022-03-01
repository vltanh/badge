import torch
from torch import nn
from torch.nn import functional as F


class MLPClassifier(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.fc = nn.Linear(512, nclasses)

    def forward(self, x):
        return self.fc(x), None


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
