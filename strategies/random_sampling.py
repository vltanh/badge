import numpy as np

from .strategy import BaseStrategy


class RandomSampling(BaseStrategy):
    def __init__(self, X, Y, net, handler, args):
        super().__init__(X, Y, net, handler, args)

    def query(self, n):
        inds = np.where(self.idxs_lb == 0)[0]
        return inds[np.random.permutation(len(inds))][:n]
