import torch as tt
import numpy as np
from torch.autograd import Function, Variable


def mean_on_023dim(x):
    return x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)


class HistoricalNormalization(Function):
    @staticmethod
    def forward(ctx, x, miu, sigma, eps=1e-5):
        ctx.multiplier = 1.0 / (sigma + eps)
        return (x - miu) * ctx.multiplier

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs[0] * ctx.multiplier, None, None, None


historical_normalization = HistoricalNormalization.apply

if __name__ == '__main__':
    from util.io import Toolkit
    from torch.autograd import gradcheck

    tk = Toolkit()
    HN = HistoricalNormalization()
    hn_foo = HN.apply

    x = Variable(tt.normal(tt.ones(1, 1, 5, 2), 2.0 * tt.ones(1, 1, 5, 2)).type(tt.DoubleTensor)
                 , requires_grad=True)

    # print(x)
    # a = hn_foo(x, 0, 1).mean()
    # a.backward()
    # a.backward(tt.ones(3, 10, 4, 4))

    # print((x - hn_foo(x, 0, 1)).max())

    check_result = gradcheck(hn_foo, (x, -18, 28989), raise_exception=True)
    print(check_result)
