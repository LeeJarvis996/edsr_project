from mindspore.nn.cell import Cell
from .basic import Dense, Dropout
from mindspore.nn import Conv1d
import math
from mindspore.ops import Exp
from mindspore.ops import Sin
from mindspore.ops import Cos
from mindspore.ops import unsqueeze
from mindspore import Parameter
from .basic import _Linear
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

class TokenEmbedding(Cell):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, pad_mode='pad', has_bias=False,
                                weight_init = "HeUniform")


    def construct(self, x):
        x = self.tokenConv(x.transpose(0, 2, 1).astype('Float32')).transpose(0, 2, 1)
        return x

class PositionalEmbedding(Cell):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = Parameter(mindspore.numpy.zeros((max_len, d_model), mindspore.float32), requires_grad=False)

        arange_tensor = mindspore.numpy.arange(0, max_len)
        tensor = Tensor(arange_tensor)
        position = tensor.unsqueeze(1).astype(mindspore.float32)

        arange_tensor = mindspore.numpy.arange(0, d_model, 2)
        arange_tensor = arange_tensor.astype(mindspore.float32)
        multiplier = -(math.log(10000.0) / d_model)
        arange_tensor = arange_tensor * multiplier
        div_term = Exp()(arange_tensor)

        pe[:, 0::2] = Sin()(position * div_term)
        pe[:, 1::2] = Cos()(position * div_term)

        pe = unsqueeze(pe, dim=0)
        self.pe = Parameter(pe, requires_grad=False)

    def construct(self, x):
        return self.pe[:, :x.shape[1]]

class TimeFeatureEmbedding(Cell):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = _Linear(d_inp, d_model, has_bias=False)

    def construct(self, x):
        # print("self.embed(x)",self.embed(x).shape)
        return self.embed(x.astype(mindspore.float32))

class DataEmbedding(Cell):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = Dropout(p=dropout)

    def construct(self, x, x_mark):
        x = x.astype(mindspore.float32)
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x_mark = x_mark.astype(mindspore.float32)
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(Cell):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = Dropout(p=dropout)
    def construct(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(Cell):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = mindspore.nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = _Linear(patch_len, d_model, has_bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)
        # Residual dropout
        self.dropout = Dropout(p=dropout)

    def construct(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = ops.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars