import numpy as np

from .strategy import BaseStrategy


def init_centers(X, K):
    # If take all (avoid errors)
    if len(X) == K:
        return np.arange(len(X))

    # List of indices
    indsAll = np.empty(K, dtype=np.uint32)

    # Initial center = point with the largest norm
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    indsAll[0] = ind

    D2 = ((X - X[ind]) ** 2).sum(-1)
    for i in range(1, K):
        Ddist = D2 / D2.sum()

        ind = np.random.choice(np.arange(len(X)), p=Ddist)
        indsAll[i] = ind

        newD = ((X - X[ind]) ** 2).sum(-1)
        D2 = np.minimum(newD, D2)
    return indsAll


class BadgeSampling(BaseStrategy):
    def __init__(self, X, Y, net, handler, args):
        super().__init__(X, Y, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled]).numpy()
        chosen = init_centers(gradEmbedding, n)
        return idxs_unlabeled[chosen]
