from torch import nn
import torch.nn.functional as F


class linMod(nn.Module):
    def __init__(self, nc=1, sz=28):
        super(linMod, self).__init__()
        self.lm = nn.Linear(int(np.prod(dim)), opts.nClasses)

    def forward(self, x):
        x = x.view(-1, int(np.prod(dim)))
        out = self.lm(x)
        return out, x

    def get_embedding_dim(self):
        return int(np.prod(dim))


class mlpMod(nn.Module):
    def __init__(self, dim, embSize=256):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, opts.nClasses)

    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = F.relu(self.lm1(x))
        out = self.lm2(emb)
        return out, emb

    def get_embedding_dim(self):
        return self.embSize
