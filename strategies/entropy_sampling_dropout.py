import numpy as np
import torch

from .strategy import BaseStrategy


class EntropySamplingDropout(BaseStrategy):
    def __init__(self, X, Y, net, handler, args, n_drop=10):
        super().__init__(X, Y, net, handler, args)
        self.n_drop = n_drop

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob_dropout(
            self.X[idxs_unlabeled], self.Y[idxs_unlabeled], self.n_drop)
        log_probs = torch.log(probs)
        U = (probs*log_probs).sum(1)
        return idxs_unlabeled[U.sort()[1][:n]]
