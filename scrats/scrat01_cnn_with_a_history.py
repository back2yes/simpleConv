import torch as tt
import numpy as np
from torch import nn
from util.io import Toolkit
import torch.nn.functional as F

tk = Toolkit()


class ImageHolder(object):
    def __init__(self, data, is_cuda=False):
        super(ImageHolder, self).__init__()
        self.data = tk.to_var(data, is_cuda)
        self.history = []


if __name__ == '__main__':
    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader
    from torchvision import transforms

    ds = CIFAR10('/home/x/data/cifar10', transform=transforms.ToTensor())
    dl = DataLoader(ds, 1, shuffle=True, pin_memory=True)

    global_counter = 0
    for epoch in range(100):
        for ii, (tsr_x, tsr_y) in enumerate(dl, 0):
            var_x, var_y = tk.to_var(tsr_x), tk.to_var(tsr_y)
