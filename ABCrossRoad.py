import numpy as np
import torch as tt
from torch import nn

class Branch(object):
    def __init__(self):
        self.quick_think = lambda : None
        self.real_think = lambda : None

class Criterion(object):
    def __init__(self):
        self.quick_think = lambda : None
        self.real_think = lambda : None

class History(object):
    def __init__(self):
        self.history_selections`

class ABCross(object):
    # Go for intuition
    def __init__(self):
        self.criterions = []
        self.branches = []

    def forward(self, x):
        expectation, criterion = self.recall(x)
        choice =
