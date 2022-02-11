import numpy as np
from .strategy import BaseStrategy
import pdb


class LeastConfidence(BaseStrategy):
    def __init__(self, X, Y, net, handler, args):
        super().__init__(X, Y, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled])
        U = probs.max(1)[0]
        return idxs_unlabeled[U.sort()[1][:n]]
