import torch as tt
import numpy as np
from torch import nn
from util.io import Toolkit
import torch.nn.functional as F
from torch.autograd import Variable


class SeparableConvolution(nn.Conv3d):
    def __init__(self, in_groups, num_filter, kernel_size=3, kernel_hsize=1, stride=1, padding=1, hpadding=0):
        super(SeparableConvolution, self).__init__(in_groups,
                                                   num_filter,
                                                   kernel_size=(kernel_hsize, kernel_size, kernel_size),
                                                   stride=(1, stride, stride),
                                                   padding=(hpadding, padding, padding),
                                                   bias=False)

    def forward(self, input):
        return F.conv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


if __name__ == '__main__':
    conv3d = SeparableConvolution(1, 1, 3)
    # print(conv3d.weight)
    x = Variable(tt.ones(4, 3, 16, 16)).unsqueeze(1)
    y = conv3d(x)
    print(y.size())
