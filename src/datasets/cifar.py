import torch
import torchvision.datasets

import constants

from . import util

def get(download=0, resize=None):
    
    download = int(download)
    
    CLASSES = 10
    CHANNELS = 3
    IMAGESIZE = (32, 32) if resize is None else resize
    
    train = torchvision.datasets.CIFAR10(root=constants.DATA_ROOT, train=True, download=download)
    data_X, data_Y = util.pillow_to_tensor(train, resize)

    test = torchvision.datasets.CIFAR10(root=constants.DATA_ROOT, train=False, download=download)
    test_X, test_Y = util.pillow_to_tensor(test, resize)

    return data_X, data_Y, test_X, test_Y, CLASSES, CHANNELS, IMAGESIZE
