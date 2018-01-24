import torch as tt
from torch import nn
from function.HistoricalNormalizationFunction import historical_normalization, mean_on_023dim


class HistoricalNormalization(nn.Module):
    def __init__(self, num_feats, momentum=0.1, eps=1e-5, clip_trivial_thres=None):
        super(HistoricalNormalization, self).__init__()
        self.running_mean = nn.Parameter(tt.zeros(1, num_feats, 1, 1))
        self.running_stdv = nn.Parameter(tt.ones(1, num_feats, 1, 1))
        self.momentum = momentum
        self.eps = eps
        self.clip_trivial_thres = clip_trivial_thres

    def forward(self, x):
        ret_val = historical_normalization(x, self.running_mean, self.running_stdv, self.eps)
        x_mean = mean_on_023dim(x)
        x_stdv = tt.sqrt(mean_on_023dim((x - x_mean) * (x - x_mean)))
        self.running_mean.data = ((1.0 - self.momentum) * self.running_mean + self.momentum * x_mean).data
        self.running_stdv.data = ((1.0 - self.momentum) * self.running_stdv + self.momentum * x_stdv).data
        return ret_val


if __name__ == '__main__':
    from torch.autograd import Function, Variable

    HN = HistoricalNormalization(3)
    x = Variable(tt.zeros(1000, 3, 8, 8).normal_(2.0, 3.0), requires_grad=True)


    print(x.mean(), x.std())

    for ii in range(30):
        y = HN(x)
        print('{:.4f}, {:.4f}'.format(y.mean().data[0], y.std().data[0]))
