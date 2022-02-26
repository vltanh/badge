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
        self.train_handler, self.test_handler = handler
        self.nclasses = len(np.unique(self.Y))

        self.idxs_lb = np.zeros(len(Y), dtype=np.bool8)

        self.clf = net.cuda()
        self.initial_state = deepcopy(net.state_dict())

        self.args = args
        self.n_pool = len(Y)

    def init_query(self, n):
        inds = np.where(self.idxs_lb == 0)[0]
        return inds[np.random.permutation(len(inds))][:n]

    def query(self, n):
        raise Exception("Not implemented")

    def update(self, idxs_lb):
        self.idxs_lb[idxs_lb] = True

    def _train(self, clf, epoch, loader_tr, optimizer):
        clf.train()
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = Variable(x.cuda()), Variable(y.cuda())
            optimizer.zero_grad()
            out, e1 = clf(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            clf.eval()
            accFinal = 0.
            for batch_idx, (x, y, idxs) in enumerate(loader_tr):
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = clf(x)
                accFinal += (torch.max(out, 1)[1] == y).sum().detach().item()
        return accFinal / len(loader_tr.dataset.X)

    def setup_network(self):
        self.clf.load_state_dict(self.initial_state)

    def setup_optimizer(self):
        if self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(self.clf.parameters(),
                                   **self.args['optimizer_args'])
        elif self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self.clf.parameters(),
                                  **self.args['optimizer_args'])
        return optimizer

    def setup_scheduler(self, optimizer):
        if self.args['scheduler'] == 'none':
            scheduler = None
        elif self.args['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args['max_epoch'])
        return scheduler

    def setup_data(self):
        X = self.X[self.idxs_lb]
        Y = self.Y[self.idxs_lb]
        ds = self.train_handler(X, Y, transform=self.args['transform'])
        return DataLoader(ds, shuffle=True, **self.args['loader_tr_args'])

    def check_saturation(self, acc_monitor):
        for i in range(len(acc_monitor)):
            for j in range(i+1, len(acc_monitor)):
                if acc_monitor[j] - acc_monitor[i] >= 0.001:
                    return False
        return True

    def train(self, dataloader, optimizer, scheduler):
        epoch = 0
        accCurrent = 0.
        acc_monitor = []

        saturated = False
        while accCurrent < self.args['max_acc'] and not saturated and epoch < self.args['max_epoch']:
            accCurrent = self._train(self.clf, epoch, dataloader, optimizer)
            if scheduler is not None:
                scheduler.step()
            print(f'Epoch {epoch}: {accCurrent}', flush=True)
            epoch += 1
            acc_monitor.append(accCurrent)
            if len(acc_monitor) >= 10:
                saturated = self.check_saturation(acc_monitor)
                del acc_monitor[0]
        return epoch

    def predict(self, X):
        loader_te = DataLoader(self.test_handler(X, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        P = torch.zeros(len(X)).long()
        with torch.no_grad():
            for x, idxs in loader_te:
                x = Variable(x.cuda())
                out, _ = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.data.cpu()
        return P

    def predict_prob(self, X):
        loader_te = DataLoader(self.test_handler(X, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        probs = torch.zeros([len(X), self.nclasses])
        with torch.no_grad():
            for x, idxs in loader_te:
                x = Variable(x.cuda())
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu().data

        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(X), self.nclasses])
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

    def predict_prob_dropout_split(self, X, n_drop):
        loader_te = DataLoader(self.test_handler(X, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([n_drop, len(X), self.nclasses])
        with torch.no_grad():
            for i in range(n_drop):
                for x, idxs in loader_te:
                    x = Variable(x.cuda())
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu().data
            return probs

    def get_embedding(self, X):
        loader_te = DataLoader(self.test_handler(X, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        embedding = torch.zeros([len(X), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, idxs in loader_te:
                x = Variable(x.cuda())
                out, e1 = self.clf(x)
                embedding[idxs] = e1.data.cpu()
        return embedding

    def get_grad_embedding(self, X):
        model = self.clf
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = self.nclasses
        embedding = np.zeros([len(X), embDim * nLab])
        loader_te = DataLoader(self.test_handler(X, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])
        with torch.no_grad():
            for x, idxs in loader_te:
                x = Variable(x.cuda())
                cout, out = self.clf(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                for j in range(len(x)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c: embDim *
                                               (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c: embDim *
                                               (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)
