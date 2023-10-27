# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""
import mindspore
from mindspore.nn.cell import Cell
import mindspore.ops as ops
import numpy as np
from mindspore import Parameter
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(Cell):
    def __init__(self):
        super(mape_loss, self).__init__()

    def construct(self, insample: mindspore.Tensor, freq: int,
                forecast: mindspore.Tensor, target: mindspore.Tensor, mask: mindspore.Tensor):
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return ops.mean(ops.abs((forecast - target) * weights))


class smape_loss(Cell):
    def __init__(self):
        super(smape_loss, self).__init__()

    def construct(self, insample: mindspore.Tensor, freq: int,
                forecast: mindspore.Tensor, target: mindspore.Tensor, mask: mindspore.Tensor):
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * ops.mean(divide_no_nan(ops.abs(forecast - target),
                                          ops.abs(Parameter(forecast, requires_grad=False)) + ops.abs(Parameter(target, requires_grad=False))) * mask)


class mase_loss(Cell):
    def __init__(self):
        super(mase_loss, self).__init__()

    def construct(self, insample: mindspore.Tensor, freq: int,
                forecast: mindspore.Tensor, target: mindspore.Tensor, mask: mindspore.Tensor):
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = ops.mean(ops.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return ops.mean(ops.abs(target - forecast) * masked_masep_inv)
