import datetime
import argparse
import time

import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils import set_deterministic, set_seed
from models import ResNet18, VGG
from models.mlp import MLPClassifier, MLPDiscriminator
from datasets import get_dataset, get_handler
from datasets.cifar10 import CIFAR10_Adversarial
from strategies import *


def plot_to_tensorboard(writer, text, fig, step):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        step (int): counter usually specifying steps/epochs/time.
    """

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range
    img = img / 255.0
    img = img.transpose(2, 0, 1)

    # Add figure in numpy "image" to TensorBoard writer
    writer.add_image(text, img, step)
    plt.close(fig)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--alg', type=str, default='rand',
    help='acquisition algorithm',
)
parser.add_argument(
    '--lr', type=float, default=1e-3,
    help='learning rate',
)
parser.add_argument(
    '--model', type=str, default='vgg',
    help='model - resnet, vgg',
)
parser.add_argument(
    '--optimizer', type=str, default='adam',
    help='optimizer - adam, sgd',
)
parser.add_argument(
    '--scheduler', type=str, default='none',
    help='scheduler - none, cosine',
)
parser.add_argument(
    '--path', type=str, default='data',
    help='data path',
)
parser.add_argument(
    '--out', type=str, default='runs',
    help='output path',
)
parser.add_argument(
    '--data', type=str,
    help='dataset (non-openML)',
)
parser.add_argument(
    '--nStart', type=int, default=1000,
    help='number of points in the beginning',
)
parser.add_argument(
    '--nQuery', type=int, default=100,
    help='number of points to query in a batch',
)
parser.add_argument(
    '--nEnd', type=int, default=25000,
    help='number of points to query in total',
)
parser.add_argument(
    '--batch_size', type=int, default=20,
    help='training batch size',
)
parser.add_argument(
    '--seed', type=int, default=3698,
    help='',
)
parser.add_argument(
    '--aug', action='store_true',
    help='',
)

opts = parser.parse_args()

SEED = opts.seed
set_deterministic()

# parameters
NUM_QUERY = opts.nQuery
DATA_NAME = opts.data

args_pool = {
    'MNIST': {
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
    },
    'FashionMNIST': {
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
    },
    'SVHN': {
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
    },
    'CIFAR10': {
        'num_class': 10,
        'transform': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ]) if opts.aug else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ]),
        'transformTest': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ]),
        'loader_tr_args': {'batch_size': opts.batch_size, 'num_workers': 0},
        'loader_te_args': {'batch_size': 1000, 'num_workers': 0},
    },
    'CIFAR100': {
        'num_class': 100,
        'transform': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ]) if opts.aug else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ]),
        'transformTest': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ]),
        'loader_tr_args': {'batch_size': opts.batch_size, 'num_workers': 0},
        'loader_te_args': {'batch_size': 1000, 'num_workers': 0},
    },
    'FOOD200': {
        'num_class': 200,
        'transform': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ]),
        'transformTest': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ]),
        'loader_tr_args': {'batch_size': opts.batch_size, 'num_workers': 0},
        'loader_te_args': {'batch_size': 128, 'num_workers': 0},
    },
}

args = args_pool[DATA_NAME]

# load specified network
set_seed(SEED)
if opts.model == 'resnet':
    net = ResNet18(nclasses=args['num_class'])
elif opts.model == 'vgg':
    net = VGG('VGG16', nclasses=args['num_class'])
else:
    raise ValueError('Invalid choice of model [resnet/vgg].')

# Load data
X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME, opts.path)
handler = get_handler(opts.data)

args['optimizer'] = opts.optimizer
if opts.optimizer == 'sgd':
    args['optimizer_args'] = {'lr': opts.lr,
                              'momentum': 0.9, 'weight_decay': 5e-4}
else:
    args['optimizer_args'] = {'lr': opts.lr, 'weight_decay': 0.0}
args['scheduler'] = opts.scheduler

args['max_epoch'] = 300
args['max_acc'] = 0.99

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)

print('Dataset:', DATA_NAME, flush=True)
print('Size of training pool: {}'.format(n_pool), flush=True)
print('Size of testing pool: {}'.format(n_test), flush=True)
print('---------------------------', flush=True)

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
elif opts.alg == 'kdpp':
    # badge but with k-DPP sampling instead of k-means++
    strategy = kDPPSampling(X_tr, Y_tr, net, handler, args)
elif opts.alg == 'kmeans':
    strategy = KMeansSampling(X_tr, Y_tr, net, handler, args)
elif opts.alg == 'waal':
    handler = CIFAR10_Adversarial, handler[1]

    clf = MLPClassifier(nclasses=args['num_class'])
    clf.fc.load_state_dict(net.classifier.state_dict())
    net = net, clf, MLPDiscriminator()

    args['alpha'] = 2e-3

    strategy = WassersteinAdversarial(X_tr, Y_tr, net, handler, args)
elif opts.alg == 'alpha_mix':
    args['alpha_closed_form_approx'] = False
    args['alpha_cap'] = 0.03125
    args['alpha_opt'] = False
    strategy = AlphaMixSampling(X_tr, Y_tr, net, handler, args)
elif opts.alg == 'albl':  # active learning by learning
    albl_list = [
        LeastConfidence(X_tr, Y_tr, net, handler, args),
        CoreSet(X_tr, Y_tr, net, handler, args)
    ]
    strategy = ActiveLearningByLearning(X_tr, Y_tr, net, handler, args,
                                        strategy_list=albl_list, delta=0.1)
else:
    raise ValueError('Invalid strategy.')

print('Strategy:', type(strategy).__name__, flush=True)
print('Query size:', NUM_QUERY, flush=True)
print('---------------------------', flush=True)

date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
writer = SummaryWriter(
    f'{opts.out}/{type(strategy).__name__}_{opts.data}_{opts.model}_{opts.optimizer}_{opts.scheduler}{"_aug" if opts.aug else ""}_{opts.batch_size}_{date}')

# Evaluate initial accuracy (without training)
P = strategy.predict(X_te)
acc = 1.0 * (Y_te == P).sum().item() / len(Y_te)
print('Test Accuracy before Training:', acc, flush=True)

# Logging
writer.add_scalar('Test Accuracy', acc, sum(strategy.idxs_lb))

# Seed
set_seed(SEED)
q_idxs = strategy.init_query(opts.nStart)
strategy.update(q_idxs)

# Train
set_seed(SEED)
strategy.setup_network()

set_seed(SEED)
optimizer = strategy.setup_optimizer()

set_seed(SEED)
scheduler = strategy.setup_scheduler(optimizer)

set_seed(SEED)
dataloader = strategy.setup_data()

set_seed(SEED)
start = time.time()
train_step = strategy.train(dataloader, optimizer, scheduler)
train_t = time.time() - start

# Evaluate
P = strategy.predict(X_te)
acc = 1.0 * (Y_te == P).sum().item() / len(Y_te)

# Logging
writer.add_scalar('Test Accuracy', acc, sum(strategy.idxs_lb))

pbar = tqdm(range(1, (opts.nEnd - opts.nStart) // NUM_QUERY + 1))
for rd in pbar:
    # Check done
    if sum(~strategy.idxs_lb) < opts.nQuery:
        print('Too few remaining points to query!', flush=True)
        continue

    # Query
    set_seed(SEED + rd)

    start = time.time()
    q_idxs = strategy.query(NUM_QUERY)
    query_t = time.time() - start

    strategy.update(q_idxs)

    # Train
    set_seed(SEED)
    strategy.setup_network()

    set_seed(SEED)
    optimizer = strategy.setup_optimizer()

    set_seed(SEED)
    scheduler = strategy.setup_scheduler(optimizer)

    set_seed(SEED)
    dataloader = strategy.setup_data()

    set_seed(SEED + rd)
    start = time.time()
    train_step = strategy.train(dataloader, optimizer, scheduler)
    train_t = time.time() - start

    # Evaluate
    P = strategy.predict(X_te)
    acc = 1.0 * (Y_te == P).sum().item() / len(Y_te)

    # Logging
    pool_size = sum(strategy.idxs_lb)
    writer.add_scalar('Query Time', query_t, pool_size)
    writer.add_scalar('Training Steps', train_step * pool_size, pool_size)
    writer.add_scalar('Training Time', train_t, pool_size)
    writer.add_scalar('Test Accuracy', acc, pool_size)

    # Plot class distribution of the pool
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=50)
    sns.countplot(x=strategy.Y[strategy.idxs_lb].numpy(), ax=ax)
    plot_to_tensorboard(writer, 'total_class_distribution', fig, rd)

    # Plot class distribution of the batch
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=50)
    sns.countplot(x=strategy.Y[q_idxs].numpy(), ax=ax)
    plot_to_tensorboard(writer, 'batch_class_distribution', fig, rd)

    pbar.set_description_str(
        f'[Round {rd:3d}] '
        + f'Query time: {query_t:.04f} | '
        + f'Training time: {train_t:.04f} | '
        + f'Test accuracy: {acc:.04f}'
    )
writer.close()
