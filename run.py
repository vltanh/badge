import os
import sys
import argparse

import numpy as np
import torch
from torchvision import transforms

import vgg
import resnet
from dataset import get_dataset, get_handler

from query_strategies import RandomSampling, BadgeSampling, \
    BaselineSampling, LeastConfidence, MarginSampling, \
    EntropySampling, CoreSet, ActiveLearningByLearning, \
    LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
    KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
    AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning

# code based on https://github.com/ej0cl6/deep-active-learning"
parser = argparse.ArgumentParser()
parser.add_argument(
    '--alg', type=str, default='rand',
    help='acquisition algorithm',
)
parser.add_argument(
    '--lr', type=float, default=1e-4,
    help='learning rate',
)
parser.add_argument(
    '--model', type=str, default='vgg',
    help='model - resnet, vgg',
)
parser.add_argument(
    '--path', type=str, default='data',
    help='data path',
)
parser.add_argument(
    '--data', type=str,
    help='dataset (non-openML)',
)
parser.add_argument(
    '--nQuery', type=int, default=100,
    help='number of points to query in a batch',
)
parser.add_argument(
    '--nStart', type=int, default=100,
    help='number of points to start',
)
parser.add_argument(
    '--nEnd', type=int, default=50000,
    help='total number of points to query',
)
parser.add_argument(
    '--nEmb', type=int, default=256,
    help='number of embedding dims (mlp)',
)
opts = parser.parse_args()

# parameters
NUM_INIT_LB = opts.nStart
NUM_QUERY = opts.nQuery
NUM_ROUND = int((opts.nEnd - NUM_INIT_LB) / opts.nQuery)
DATA_NAME = opts.data

args_pool = {
    'MNIST': {
        'n_epoch': 10,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'transformTest': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
        'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
        'optimizer_args': {'lr': 0.01, 'momentum': 0.5},
    },
    'FashionMNIST': {
        'n_epoch': 10,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'transformTest': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
        'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
        'optimizer_args': {'lr': 0.01, 'momentum': 0.5},
    },
    'SVHN': {
        'n_epoch': 20,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                 (0.1980, 0.2010, 0.1970))
        ]),
        'transformTest': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                 (0.1980, 0.2010, 0.1970))
        ]),
        'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
        'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
        'optimizer_args': {'lr': 0.01, 'momentum': 0.5},
    },
    'CIFAR10': {
        'n_epoch': 3,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ]),
        'transformTest': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ]),
        'loader_tr_args': {'batch_size': 128, 'num_workers': 1},
        'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
        'optimizer_args': {'lr': 0.05, 'momentum': 0.3},
    }
}

opts.nClasses = 10

args = args_pool[DATA_NAME]
if not os.path.exists(opts.path):
    os.makedirs(opts.path)

X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME, opts.path)
opts.dim = np.shape(X_tr)[1:]
handler = get_handler(opts.data)

args['lr'] = opts.lr

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(NUM_INIT_LB), flush=True)
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB), flush=True)
print('number of testing pool: {}'.format(n_test), flush=True)

# generate initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True


# load specified network
if opts.model == 'resnet':
    net = resnet.ResNet18()
elif opts.model == 'vgg':
    net = vgg.VGG('VGG16')
else:
    print('choose a valid model - resnet, or vgg', flush=True)
    raise ValueError

if type(X_tr[0]) is not np.ndarray:
    X_tr = X_tr.numpy()

# set up the specified sampler
if opts.alg == 'rand':  # random sampling
    strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'conf':  # confidence-based sampling
    strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'marg':  # margin-based sampling
    strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'badge':  # batch active learning by diverse gradient embeddings
    strategy = BadgeSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'coreset':  # coreset sampling
    strategy = CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'entropy':  # entropy-based sampling
    strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'baseline':
    # badge but with k-DPP sampling instead of k-means++
    strategy = BaselineSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'albl':  # active learning by learning
    albl_list = [
        LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args),
        CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)
    ]
    strategy = ActiveLearningByLearning(X_tr, Y_tr,
                                        idxs_lb, net, handler, args,
                                        strategy_list=albl_list, delta=0.1)
else:
    print('choose a valid acquisition function', flush=True)
    raise ValueError

# print info
print(DATA_NAME, flush=True)
print(type(strategy).__name__, flush=True)

# round 0 accuracy
strategy.train()
P = strategy.predict(X_te, Y_te)
acc = np.zeros(NUM_ROUND+1)
acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
print(str(opts.nStart) + '\ttesting accuracy {}'.format(acc[0]), flush=True)

for rd in range(1, NUM_ROUND+1):
    print('Round {}'.format(rd), flush=True)

    # query
    output = strategy.query(NUM_QUERY)
    q_idxs = output
    idxs_lb[q_idxs] = True

    # report weighted accuracy
    corr = strategy.predict(
        X_tr[q_idxs],
        torch.Tensor(Y_tr.numpy()[q_idxs]).long()
    ).numpy() == Y_tr.numpy()[q_idxs]

    # update
    strategy.update(idxs_lb)
    strategy.train()

    # round accuracy
    P = strategy.predict(X_te, Y_te)
    acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    print(str(sum(idxs_lb)) + '\t' +
          'testing accuracy {}'.format(acc[rd]), flush=True)
    if sum(~strategy.idxs_lb) < opts.nQuery:
        sys.exit('too few remaining points to query')
