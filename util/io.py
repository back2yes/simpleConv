import torch as tt
import numpy as np
from torch.autograd import Variable

class Toolkit(object):
    def __init__(self, viz_dir=None):
        super(Toolkit, self).__init__()
        self.viz_dir=viz_dir

    @staticmethod
    def to_var(x, is_cuda=True):
        return Variable(x).cuda() if is_cuda else Variable(x)
