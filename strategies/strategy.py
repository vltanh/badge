import numpy as np
import torch
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from copy import deepcopy


class BaseStrategy:
    def __init__(self, X, Y, net, handler, args):
        self.X = X
        self.Y = Y
        self.handler = handler

        self.idxs_lb = np.zeros(len(Y), dtype=np.bool8)

        self.clf = net.cuda()
        self.initial_state = net.state_dict()

        self.args = args
        self.n_pool = len(Y)

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb[idxs_lb] = True

    def _train(self, clf, epoch, loader_tr, optimizer):
        clf.train()
        accFinal = 0.
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = Variable(x.cuda()), Variable(y.cuda())
            optimizer.zero_grad()
            out, e1 = clf(x)
            loss = F.cross_entropy(out, y)
            accFinal += (torch.max(out, 1)[1] == y).sum().detach().item()
            loss.backward()
            optimizer.step()
        return accFinal / len(loader_tr.dataset.X)

    def train(self):
        # Load initial state
        clf = self.clf.cuda()
        clf.load_state_dict(self.initial_state)

        # Load optimizer
        optimizer = optim.Adam(clf.parameters(),
                               lr=self.args['lr'], weight_decay=0)

        # Prepare training data
        X = self.X[self.idxs_lb]
        Y = self.Y[self.idxs_lb]
        ds = self.handler(X, Y, transform=self.args['transform'])
        dl = DataLoader(ds, shuffle=True, **self.args['loader_tr_args'])

        # Train until converge
        epoch = 1
        accCurrent = 0.
        while accCurrent < 0.99:
            accCurrent = self._train(clf, epoch, dl, optimizer)
            epoch += 1

        self.clf = clf

    def predict(self, X, Y):
        if type(X) is np.ndarray:
            loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                                   shuffle=False, **self.args['loader_te_args'])
        else:
            loader_te = DataLoader(self.handler(X.numpy(), Y, transform=self.args['transformTest']),
                                   shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        P = torch.zeros(len(Y)).long()
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.data.cpu()
        return P

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu().data

        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu().data
        probs /= n_drop

        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu().data
            return probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                embedding[idxs] = e1.data.cpu()

        return embedding

    # gradient embedding (assumes cross-entropy loss)
    def get_grad_embedding(self, X, Y):
        model = self.clf
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                cout, out = self.clf(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c: embDim *
                                               (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c: embDim *
                                               (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)
