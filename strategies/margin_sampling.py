import numpy as np

from .strategy import BaseStrategy


class MarginSampling(BaseStrategy):
    def __init__(self, X, Y, net, handler, args):
        super().__init__(X, Y, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:, 1]
        return idxs_unlabeled[U.sort()[1].numpy()[:n]]
