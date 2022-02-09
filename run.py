import datetime
import argparse
import time

import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import set_deterministic, set_seed
from models import ResNet18, VGG
from getter import get_dataset, get_handler
from strategies import RandomSampling, BadgeSampling, \
    BaselineSampling, LeastConfidence, MarginSampling, \
    EntropySampling, CoreSet, ActiveLearningByLearning, \
    LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
    KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
    AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning, WassersteinAdversarial

SEED = 3698
set_deterministic()

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

# load specified network
set_seed(SEED)
if opts.model == 'resnet':
    net = ResNet18()
elif opts.model == 'vgg':
    net = VGG('VGG16')
else:
    raise ValueError('Invalid choice of model [resnet/vgg].')

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
        'loader_tr_args': {'batch_size': 64, 'num_workers': 0},
        'loader_te_args': {'batch_size': 1000, 'num_workers': 0},
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
        'loader_tr_args': {'batch_size': 64, 'num_workers': 0},
        'loader_te_args': {'batch_size': 1000, 'num_workers': 0},
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
        'loader_tr_args': {'batch_size': 64, 'num_workers': 0},
        'loader_te_args': {'batch_size': 1000, 'num_workers': 0},
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
        'loader_tr_args': {'batch_size': 128, 'num_workers': 0},
        'loader_te_args': {'batch_size': 1000, 'num_workers': 0},
        'optimizer_args': {'lr': 0.05, 'momentum': 0.3},
    }
}

args = args_pool[DATA_NAME]
X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME, opts.path)
handler = get_handler(opts.data)

args['lr'] = opts.lr

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)

print('Dataset:', DATA_NAME)
print('Size of training pool: {}'.format(n_pool))
print('Size of testing pool: {}'.format(n_test))
print('---------------------------')

# set up the specified sampler
set_seed(SEED)
if opts.alg == 'rand':  # random sampling
    strategy = RandomSampling(X_tr, Y_tr, net, handler, args)
elif opts.alg == 'conf':  # confidence-based sampling
    strategy = LeastConfidence(X_tr, Y_tr, net, handler, args)
elif opts.alg == 'marg':  # margin-based sampling
    strategy = MarginSampling(X_tr, Y_tr, net, handler, args)
elif opts.alg == 'badge':  # batch active learning by diverse gradient embeddings
    strategy = BadgeSampling(X_tr, Y_tr, net, handler, args)
elif opts.alg == 'coreset':  # coreset sampling
    strategy = CoreSet(X_tr, Y_tr, net, handler, args)
elif opts.alg == 'entropy':  # entropy-based sampling
    strategy = EntropySampling(X_tr, Y_tr, net, handler, args)
elif opts.alg == 'baseline':
    # badge but with k-DPP sampling instead of k-means++
    strategy = BaselineSampling(X_tr, Y_tr, net, handler, args)
elif opts.alg == 'was_adv':
    strategy = WassersteinAdversarial
elif opts.alg == 'albl':  # active learning by learning
    albl_list = [
        LeastConfidence(X_tr, Y_tr, net, handler, args),
        CoreSet(X_tr, Y_tr, net, handler, args)
    ]
    strategy = ActiveLearningByLearning(X_tr, Y_tr,
                                        net, handler, args,
                                        strategy_list=albl_list, delta=0.1)
else:
    raise ValueError('Invalid strategy.')

print('Strategy:', type(strategy).__name__)
print('Query size:', NUM_QUERY)
print('---------------------------')


date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
writer = SummaryWriter(f'runs/{type(strategy).__name__}_{DATA_NAME}_{date}')
acc = np.zeros(NUM_ROUND+1)
query_times = []

# round 0 accuracy
# print(f'Round 0')

# Generate initial pool
# set_seed(SEED)
# q_idxs = np.random.choice(np.arange(n_pool), size=NUM_INIT_LB)
# strategy.update(q_idxs)

# Train
# set_seed(SEED)
# strategy.train()

# Evaluate
P = strategy.predict(X_te, Y_te)
acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)

# Report
# print('Size of labeled pool:', sum(strategy.idxs_lb))
# print('Test Accuracy:', acc[0])
# print('===')
writer.add_scalar('Test Accuracy', acc[0], sum(strategy.idxs_lb))

pbar = tqdm(range(1, NUM_ROUND+1))
for rd in pbar:
    # print(f'Round {rd}')

    # Query
    set_seed(SEED)
    start = time.time()
    q_idxs = strategy.query(NUM_QUERY)
    query_times.append(time.time() - start)
    # print('Query time:', time.time() - start)

    # Train
    set_seed(SEED)
    strategy.update(q_idxs)
    strategy.train()

    # Evaluate
    P = strategy.predict(X_te, Y_te)
    acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)

    # Report
    # print('Size of labeled pool:', sum(strategy.idxs_lb))
    # print('Test Accuracy:', acc[rd])
    # print('===')
    writer.add_scalar('Test Accuracy', acc[rd], sum(strategy.idxs_lb))

    # Check done
    if sum(~strategy.idxs_lb) < opts.nQuery:
        print('Too few remaining points to query!')
        break

    pbar.set_description_str(
        f'[Round {rd:3d}] Query time: {np.mean(query_times):.02f} +/- {np.std(query_times):.04f}'
    )
