from typing import Union, Optional
import mindspore
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, XavierNormal, XavierUniform, \
    HeUniform, Uniform, _calculate_fan_in_and_fan_out
from mindspore.nn.cell import Cell
from .basic import _Linear, Dropout
import math
import numpy as np
from scipy.fftpack import next_fast_len
from layer.container import CellList


class scale_block(Cell):
    def __init__(self, configs, win_size, d_model, n_heads, d_ff, depth, dropout, \
                 seg_num=10, factor=10):
        super(scale_block, self).__init__()
        if win_size > 1:
            self.merge_layer = SegMerging(d_model, win_size, mindspore.nn.LayerNorm)
        else:
            self.merge_layer = None
        # self.encode_layers = nn.ModuleList()
        self.encode_layers = CellList()
        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(configs, seg_num, factor, d_model, n_heads, \
                                                             d_ff, dropout))

    def construct(self, x, attn_mask=None, tau=None, delta=None):
        _, ts_dim, _, _ = x.shape
        if self.merge_layer is not None:
            x = self.merge_layer(x)
        for layer in self.encode_layers:
            x = layer(x)
        return x, None

class SegMerging(Cell):
    def __init__(self, d_model, win_size, norm_layer=mindspore.nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = _Linear(win_size * d_model, d_model)
        # self.norm = norm_layer(win_size * d_model)
        self.norm = norm_layer((win_size * d_model,))
    def construct(self, x):
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            # x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)
            x = ops.concat((x, x[:, :, -pad_num:, :]), -2)
        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        # x = torch.cat(seg_to_merge, -1)
        x = ops.concat(seg_to_merge, -1)
        x = self.norm(x)
        x = self.linear_trans(x)
        return x