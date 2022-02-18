import os

from .svhn import *
from .mnist import *
from .fashion_mnist import *
from .cifar10 import *
from .cifar100 import *
from .food200 import *


def get_dataset(name, path):
    if not os.path.exists(path):
        os.makedirs(path)

    if name == 'MNIST':
        return get_MNIST(path)
    elif name == 'FashionMNIST':
        return get_FashionMNIST(path)
    elif name == 'SVHN':
        return get_SVHN(path)
    elif name == 'CIFAR10':
        return get_CIFAR10(path)
    elif name == 'CIFAR100':
        return get_CIFAR100(path)
    elif name == 'FOOD200':
        return get_FOOD200(path)
    else:
        raise ValueError('Invalid data name.')


def get_handler(name):
    if name == 'MNIST':
        return DataHandler3
    elif name == 'FashionMNIST':
        return DataHandler1
    elif name == 'SVHN':
        return SVHN_Labeled, SVHN_Unlabeled
    elif name == 'CIFAR10':
        return CIFAR10_Labeled, CIFAR10_Unlabeled
    elif name == 'CIFAR100':
        return CIFAR100_Labeled, CIFAR100_Unlabeled
    elif name == 'FOOD200':
        return FOOD200_Labeled, FOOD200_Unlabeled
    else:
        raise ValueError('Invalid data name.')
