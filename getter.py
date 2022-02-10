
import os

from datasets import *


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
    else:
        raise ValueError('Invalid data name.')


def get_handler(name):
    if name == 'MNIST':
        return DataHandler3
    elif name == 'FashionMNIST':
        return DataHandler1
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'CIFAR10':
        return CIFAR10Train, CIFAR10Test
    else:
        raise ValueError('Invalid data name.')