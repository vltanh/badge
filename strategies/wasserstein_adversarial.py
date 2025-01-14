from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import grad

from .strategy import BaseStrategy


# setting gradient values
def set_requires_grad(model, requires_grad=True):
    """
    Used in training adversarial approach
    :param model:
    :param requires_grad:
    :return:
    """

    for param in model.parameters():
        param.requires_grad = requires_grad


# setting gradient penalty for sure the lipschitz property
def gradient_penalty(critic, h_s, h_t):
    ''' Gradient penalty approach'''
    alpha = torch.rand(h_s.size(0), 1).cuda()
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
    # interpolates.requires_grad_()
    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()

    return gradient_penalty


class WassersteinAdversarial(BaseStrategy):
    def __init__(self, X, Y, net, handler, args):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.X = X
        self.Y = Y

        self.fea, self.clf, self.dis = net
        self.fea = self.fea.to(self.device)
        self.clf = self.clf.to(self.device)
        self.dis = self.dis.to(self.device)

        self.init_fea = deepcopy(self.fea.state_dict())
        self.init_clf = deepcopy(self.clf.state_dict())
        self.init_dis = deepcopy(self.dis.state_dict())

        self.train_handler, self.test_handler = handler
        self.args = args

        self.n_pool = len(Y)
        self.num_class = self.args['num_class']

        self.selection = 10
        # for cifar 10 or svhn or fashion mnist  self.selection = 10

        self.idxs_lb = np.zeros(len(Y), dtype=np.bool8)

    def setup_network(self):
        self.fea.load_state_dict(self.init_fea)
        self.clf.load_state_dict(self.init_clf)
        self.dis.load_state_dict(self.init_dis)

    def setup_data(self):
        idx_lb_train = np.arange(self.n_pool)[self.idxs_lb]
        idx_ulb_train = np.arange(self.n_pool)[~self.idxs_lb]
        return DataLoader(self.train_handler(self.X[idx_lb_train], self.Y[idx_lb_train],
                                             self.X[idx_ulb_train], self.Y[idx_ulb_train],
                                             transform=self.args['transform']),
                          shuffle=True, **self.args['loader_tr_args'])

    def setup_optimizer(self):
        if self.args['optimizer'] == 'adam':
            opt = optim.Adam
        else:
            opt = optim.SGD

        opt_fea = opt(self.fea.parameters(),
                      **self.args['optimizer_args'])
        opt_clf = opt(self.clf.parameters(),
                      **self.args['optimizer_args'])
        opt_dis = opt(self.dis.parameters(),
                      **self.args['optimizer_args'])
        return opt_fea, opt_clf, opt_dis

    def setup_scheduler(self, optimizer):
        if self.args['scheduler'] == 'none':
            scheduler = [None for opt in optimizer]
        elif self.args['scheduler'] == 'cosine':
            scheduler = [optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.args['max_epoch']) for opt in optimizer]
        return scheduler

    def _train(self, epoch, dataloader, optimizer):
        alpha = self.args['alpha']

        opt_fea, opt_clf, opt_dis = optimizer

        idx_lb_train = np.arange(self.n_pool)[self.idxs_lb]
        idx_ulb_train = np.arange(self.n_pool)[~self.idxs_lb]

        # computing the unbalancing ratio, a value betwwen [0,1], generally 0.1 - 0.5
        gamma_ratio = len(idx_lb_train) / len(idx_ulb_train)

        self.fea.train()
        self.clf.train()
        self.dis.train()

        for index, label_x, label_y, unlabel_x, _ in dataloader:
            label_x, label_y = label_x.cuda(), label_y.cuda()
            unlabel_x = unlabel_x.cuda()

            # training feature extractor and predictor
            set_requires_grad(self.fea, requires_grad=True)
            set_requires_grad(self.clf, requires_grad=True)
            set_requires_grad(self.dis, requires_grad=False)

            _, lb_z = self.fea(label_x)
            _, unlb_z = self.fea(unlabel_x)

            opt_fea.zero_grad()
            opt_clf.zero_grad()

            lb_out, _ = self.clf(lb_z)

            # prediction loss (deafult we use F.cross_entropy)
            pred_loss = torch.mean(F.cross_entropy(lb_out, label_y))

            # Wasserstein loss (here is the unbalanced loss, because we used the redundant trick)
            wassertein_distance = self.dis(unlb_z).mean() \
                - gamma_ratio * self.dis(lb_z).mean()

            with torch.no_grad():
                _, lb_z = self.fea(label_x)
                _, unlb_z = self.fea(unlabel_x)

            gp = gradient_penalty(self.dis, unlb_z, lb_z)

            loss = pred_loss + alpha * wassertein_distance + alpha * gp * 5
            # for CIFAR10 the gradient penality is 5
            # for SVHN the gradient penality is 2

            loss.backward()
            opt_fea.step()
            opt_clf.step()

            # Then the second step, training discriminator
            set_requires_grad(self.fea, requires_grad=False)
            set_requires_grad(self.clf, requires_grad=False)
            set_requires_grad(self.dis, requires_grad=True)

            with torch.no_grad():
                _, lb_z = self.fea(label_x)
                _, unlb_z = self.fea(unlabel_x)

            for _ in range(1):
                # gradient ascent for multiple times like GANS training
                gp = gradient_penalty(self.dis, unlb_z, lb_z)

                wassertein_distance = self.dis(unlb_z).mean() \
                    - gamma_ratio * self.dis(lb_z).mean()

                dis_loss = -1 * alpha * wassertein_distance - alpha * gp * 2

                opt_dis.zero_grad()
                dis_loss.backward()
                opt_dis.step()

    @torch.no_grad()
    def _eval(self, epoch, dataloader):
        self.fea.eval()
        self.clf.eval()
        self.dis.eval()

        n_batch = 0
        acc = 0.
        for index, label_x, label_y, _, _ in dataloader:
            n_batch += len(label_y)

            label_x, label_y = label_x.cuda(), label_y.cuda()
            _, lb_z = self.fea(label_x)
            lb_out, _ = self.clf(lb_z)

            P = lb_out.max(1)[1]
            acc += 1.0 * (label_y == P).sum().item()
        acc /= n_batch
        return acc

    def step_scheduler(self, scheduler):
        for sc in scheduler:
            if sc is not None:
                sc.step()

    def predict(self, X):
        loader_te = DataLoader(self.test_handler(X, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])
        self.fea.eval()
        self.clf.eval()
        P = torch.zeros(len(X)).long()
        with torch.no_grad():
            for x, idxs in loader_te:
                x = x.to(self.device)
                _, latent = self.fea(x)
                out, _ = self.clf(latent)
                pred = out.max(1)[1]
                P[idxs] = pred.cpu()
        return P

    def predict_prob(self, X):
        loader_te = DataLoader(self.test_handler(X, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])

        self.fea.eval()
        self.clf.eval()

        probs = torch.zeros([len(X), self.num_class])
        with torch.no_grad():

            for x, idxs in loader_te:
                x = x.to(self.device)
                _, latent = self.fea(x)
                out, _ = self.clf(latent)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()

        return probs

    def pred_dis_score(self, X):
        loader_te = DataLoader(self.test_handler(X, transform=self.args['transformTest']),
                               shuffle=False, **self.args['loader_te_args'])

        self.fea.eval()
        self.dis.eval()

        scores = torch.zeros(len(X))

        with torch.no_grad():
            for x, idxs in loader_te:
                x = x.to(self.device)
                _, latent = self.fea(x)
                out = self.dis(latent).cpu()
                scores[idxs] = out.view(-1)

        return scores

    def single_worst(self, probas):
        value, _ = torch.max(-1*torch.log(probas), 1)

        return value

    def L2_upper(self, probas):
        value = torch.norm(torch.log(probas), dim=1)

        return value

    def L1_upper(self, probas):
        value = torch.sum(-1*torch.log(probas), dim=1)

        return value

    def query(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        # prediction output probability
        probs = self.predict_prob(self.X[idxs_unlabeled])

        # uncertainly score (three options, single_worst, L2_upper, L1_upper)
        # uncertainly_score = self.single_worst(probs)
        uncertainly_score = 0.5 * self.L2_upper(probs) \
            + 0.5 * self.L1_upper(probs)

        # prediction output discriminative score
        dis_score = self.pred_dis_score(self.X[idxs_unlabeled])

        # computing the decision score
        total_score = uncertainly_score - self.selection * dis_score
        b = total_score.sort()[1][:query_num]

        # sort the score with minimal query_number examples
        # expected value outputs from smaller to large
        return idxs_unlabeled[total_score.sort()[1][:query_num]]
