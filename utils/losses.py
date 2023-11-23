import mindspore
from mindspore.nn.cell import Cell
import mindspore.ops as ops
import numpy as np
from mindspore import Parameter
import pdb


def divide_no_nan(a, b):
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(Cell):
    def __init__(self):
        super(mape_loss, self).__init__()

    def construct(self, insample: mindspore.Tensor, freq: int,
                forecast: mindspore.Tensor, target: mindspore.Tensor, mask: mindspore.Tensor):
        weights = divide_no_nan(mask, target)
        return ops.mean(ops.abs((forecast - target) * weights))


class smape_loss(Cell):
    def __init__(self):
        super(smape_loss, self).__init__()

    def construct(self, insample: mindspore.Tensor, freq: int,
                forecast: mindspore.Tensor, target: mindspore.Tensor, mask: mindspore.Tensor):
        return 200 * ops.mean(divide_no_nan(ops.abs(forecast - target),
                                          ops.abs(Parameter(forecast, requires_grad=False)) + ops.abs(Parameter(target, requires_grad=False))) * mask)


class mase_loss(Cell):
    def __init__(self):
        super(mase_loss, self).__init__()

    def construct(self, insample: mindspore.Tensor, freq: int,
                forecast: mindspore.Tensor, target: mindspore.Tensor, mask: mindspore.Tensor):
        masep = ops.mean(ops.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return ops.mean(ops.abs(target - forecast) * masked_masep_inv)
