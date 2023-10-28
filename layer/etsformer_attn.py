import mindspore
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.nn.cell import Cell
from .basic import _Linear, Dropout
import math
import numpy as np
from scipy.fftpack import next_fast_len
import torch


class Transform:
    def __init__(self, sigma):
        self.sigma = sigma
    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))
    def jitter(self, x):
        return x + (ops.randn(x.shape) * self.sigma)
    def scale(self, x):
        return x * (ops.randn(x.shape[-1]) * self.sigma + 1)
    def shift(self, x):
        return x + (ops.randn(x.shape[-1]) * self.sigma)


class GrowthLayer(Cell):
    def __init__(self, d_model, nhead, d_head=None, dropout=0.1):
        super().__init__()
        self.d_head = d_head or (d_model // nhead)
        self.d_model = d_model
        self.nhead = nhead

        # self.z0 = nn.Parameter(torch.randn(self.nhead, self.d_head))
        self.z0 = Parameter(mindspore.numpy.randn((self.nhead, self.d_head)), requires_grad=True)
        self.in_proj = _Linear(self.d_model, self.d_head * self.nhead)
        self.es = ExponentialSmoothing(self.d_head, self.nhead, dropout=dropout)
        self.out_proj = _Linear(self.d_head * self.nhead, self.d_model)
        assert self.d_head * self.nhead == self.d_model, "d_model must be divisible by nhead"

    def construct(self, inputs):
        """
        :param inputs: shape: (batch, seq_len, dim)
        :return: shape: (batch, seq_len, dim)
        """
        b, t, d = inputs.shape
        values = self.in_proj(inputs.astype(mindspore.float32)).view(b, t, self.nhead, -1)
        # values = torch.cat([repeat(self.z0, 'h d -> b 1 h d', b=b), values], dim=1)
        temp = mindspore.numpy.randn((b, self.z0.shape[0], self.z0.shape[1]))
        values = ops.concat([self.z0.expand_as(temp).expand_dims(axis = 1).astype(mindspore.float64), values.astype(mindspore.float64)], 1)
        values = values[:, 1:] - values[:, :-1]
        out = self.es(values)

        # out = torch.cat([repeat(self.es.v0, '1 1 h d -> b 1 h d', b=b), out], dim=1)
        temp = mindspore.numpy.randn((b, 1, self.es.v0.shape[2], self.es.v0.shape[3]))
        out = ops.concat([self.es.v0.expand_as(temp).astype(mindspore.float64), out.astype(mindspore.float64)], 1)
        # out = rearrange(out, 'b t h d -> b t (h d)')
        out = out.view(out.shape[0], out.shape[1], out.shape[2] * out.shape[3])
        # print(out.shape)
        return self.out_proj(out.astype(mindspore.float32))


class ExponentialSmoothing(Cell):
    def __init__(self, dim, nhead, dropout=0.1, aux=False):
        super().__init__()
        # self._smoothing_weight = nn.Parameter(torch.randn(nhead, 1))
        # self.v0 = nn.Parameter(torch.randn(1, 1, nhead, dim))
        self._smoothing_weight = Parameter(mindspore.numpy.randn((nhead, 1)), requires_grad=True)
        self.v0 = Parameter(mindspore.numpy.randn((1, 1, nhead, dim)), requires_grad=True)
        self.dropout = Dropout(p = dropout)
        if aux:
            self.aux_dropout = Dropout(p = dropout)

    def construct(self, values, aux_values=None):
        b, t, h, d = values.shape
        init_weight, weight = self.get_exponential_weight(t)
        output = conv1d_fft(self.dropout(values), weight, dim=1)
        output = init_weight * self.v0 + output
        if aux_values is not None:
            aux_weight = weight / (1 - self.weight) * self.weight
            aux_output = conv1d_fft(self.aux_dropout(aux_values), aux_weight)
            output = output + aux_output

        return output

    def get_exponential_weight(self, T):
        # Generate array [0, 1, ..., T-1]
        # powers = torch.arange(T, dtype=torch.float, device=self.weight.device)
        powers = ops.arange(T, dtype = mindspore.float64)

        # (1 - \alpha) * \alpha^t, for all t = T-1, T-2, ..., 0]
        # weight = (1 - self.weight) * (self.weight ** torch.flip(powers, dims=(0,)))
        weight = (1 - self.weight) * (self.weight ** ops.flip(powers, dims=(0,)))

        # \alpha^t for all t = 1, 2, ..., T
        init_weight = self.weight ** (powers + 1)

        # return rearrange(init_weight, 'h t -> 1 t h 1'), \
        #        rearrange(weight, 'h t -> 1 t h 1')
        return ops.swapaxes(init_weight, 0, 1).expand_dims(axis = 0).expand_dims(axis = -1), \
               ops.swapaxes(weight, 0, 1).expand_dims(axis=0).expand_dims(axis=-1)

    @property
    def weight(self):
        # return torch.sigmoid(self._smoothing_weight)
        return ops.sigmoid(self._smoothing_weight)


def conv1d_fft(f, g, dim=-1):
    # N = f.size(dim)
    # M = g.size(dim)
    N = f.shape[dim]
    M = g.shape[dim]
    fast_len = next_fast_len(N + M - 1)
    '''
    F_f = fft.rfft(f, fast_len, dim=dim)
    F_g = fft.rfft(g, fast_len, dim=dim)
    ft.irfft(F_fg, fast_len, dim=dim)
    Since the mindspore.ops.FFTWithSize is not same as the torch.rfft/irfft, we use numpy instead.
    '''
    '''
    TypeError: For 'Mul', gradient not support for complex type currently.
    We have to do multiplication with nd.array format.
    If gradient is supported in the future, please use this code:
        # F_f = Tensor(np.fft.rfft(f.asnumpy(), fast_len, axis=dim))
        # F_g = Tensor(np.fft.rfft(g.asnumpy(), fast_len, axis=dim))
        # F_fg = F_f * F_g.conj()
        # out = Tensor(np.fft.irfft(F_fg.asnumpy(), fast_len, axis=dim))
    '''
    F_f = np.fft.rfft(f.asnumpy(), fast_len, axis=dim)
    F_g = Tensor(np.fft.rfft(g.asnumpy(), fast_len, axis=dim))
    F_fg = F_f * F_g.conj().asnumpy()
    out = Tensor(np.fft.irfft(F_fg, fast_len, axis=dim))

    # out = out.roll((-1,), dims=(dim,))
    out = Tensor(np.roll(out.asnumpy(), shift=(-1,),axis=(dim,)))
    # idx = torch.as_tensor(range(fast_len - N, fast_len)).to(out.device)
    idx = mindspore.numpy.arange(start=fast_len - N, stop=fast_len)
    out = out.index_select(dim, idx)
    return out


class FourierLayer(Cell):

    def __init__(self, d_model, pred_len, k=None, low_freq=1):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq

    def construct(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        # x_freq = fft.rfft(x, dim=1)
        x_freq = Tensor(np.fft.rfft(x.asnumpy(), axis=1))
        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            # f = fft.rfftfreq(t)[self.low_freq:-1]
            f = Tensor(np.fft.rfftfreq(t)[self.low_freq:-1])
        else:
            x_freq = x_freq[:, self.low_freq:]
            # f = fft.rfftfreq(t)[self.low_freq:]
            f = Tensor(np.fft.rfftfreq(t)[self.low_freq:])
        x_freq, index_tuple = self.topk_freq(x_freq)
        # f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))
        d = f.repeat([x_freq.shape[2]], axis=0).view(f.shape[0], x_freq.shape[2])
        temp = mindspore.numpy.randn((x_freq.shape[0], f.shape[0], x_freq.shape[2]))
        f = d.expand_as(temp)
        # f = rearrange(f[index_tuple], 'b f d -> b f () d')
        f = f[index_tuple].expand_dims(axis = 2)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        # x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        # f = torch.cat([f, -f], dim=1)
        x_freq = ops.concat([x_freq, x_freq.conj()], 1)
        f = ops.concat([f, -f], 1)
        # t_val = rearrange(torch.arange(t + self.pred_len, dtype=torch.float),
        #                   't -> () () t ()').to(x_freq.device)
        t_val = ops.arange(t + self.pred_len, dtype=mindspore.float64).expand_dims(axis=0).expand_dims(axis=1).expand_dims(axis=-1)
        # amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
        # phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        amp = Tensor(x_freq.abs().asnumpy() / t, dtype = mindspore.float64)
        amp = amp.expand_dims(axis=2)
        phase = x_freq.angle().expand_dims(axis=-2)
        # x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)
        '''
        ERROR: For 'Mul', gradient not support for complex type currently.
        If gradient for complex type is supported in the future, please use this code:
        # x_time = amp * ops.cos(2 * math.pi * f * t_val + phase)
        '''
        x_time = amp.asnumpy() * ops.cos(2 * math.pi * f * t_val + phase).asnumpy()
        x_time = Tensor(x_time)
        op = ops.ReduceSum(keep_dims=True)
        # return reduce(x_time, 'b f t d -> b t d', 'sum')
        return op(x_time, 1).squeeze()

    def topk_freq(self, x_freq):
        '''
        For now, ops.topk does not support Complex input, and therefore we use Pytorch instead.
        # values, indices = ops.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        '''
        # values, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        values, indices = torch.topk(torch.from_numpy(x_freq.asnumpy()).abs(), self.k, dim=1, largest=True, sorted=True)
        values, indices = Tensor(values.numpy()), Tensor(indices.numpy())
        # mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)))
        mesh_a, mesh_b = ops.meshgrid(ops.arange(x_freq.shape[0]), ops.arange(x_freq.shape[2]), indexing='ij')
        # index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        index_tuple = (ops.ExpandDims()(mesh_a, 1), indices, ops.ExpandDims()(mesh_b, 1))
        # print(index_tuple)
        # print(x_freq.shape)
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple


class LevelLayer(Cell):
    def __init__(self, d_model, c_out, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.c_out = c_out
        self.es = ExponentialSmoothing(1, self.c_out, dropout=dropout, aux=True)
        self.growth_pred = _Linear(self.d_model, self.c_out)
        self.season_pred = _Linear(self.d_model, self.c_out)
    def construct(self, level, growth, season):
        b, t, _ = level.shape
        growth = self.growth_pred(growth).view(b, t, self.c_out, 1)
        season = self.season_pred(season.astype(mindspore.float32)).view(b, t, self.c_out, 1)
        growth = growth.view(b, t, self.c_out, 1)
        season = season.view(b, t, self.c_out, 1)
        level = level.view(b, t, self.c_out, 1)
        out = self.es(level - season, aux_values=growth)
        # out = rearrange(out, 'b t h d -> b t (h d)')
        out = out.view(out.shape[0], out.shape[1], out.shape[2] * out.shape[3])
        return out

class Feedforward(Cell):
    def __init__(self, d_model, dim_feedforward, dropout=0.1, activation='sigmoid'):
        # Implementation of Feedforward model
        super().__init__()
        self.linear1 = _Linear(d_model, dim_feedforward, has_bias=False)
        self.dropout1 = Dropout(p = dropout)
        self.linear2 = _Linear(dim_feedforward, d_model, has_bias=False)
        self.dropout2 = Dropout(p = dropout)
        self.activation = getattr(ops, activation)

    def construct(self, x):
        x = self.linear2(self.dropout1(self.activation(self.linear1(x.astype(mindspore.float32)))))
        return self.dropout2(x)


class DampingLayer(Cell):

    def __init__(self, pred_len, nhead, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.nhead = nhead
        # self._damping_factor = nn.Parameter(torch.randn(1, nhead))
        self._damping_factor = Parameter(mindspore.numpy.randn((1, nhead)), requires_grad=True)
        self.dropout = Dropout(p = dropout)
    def construct(self, x):
        # x = repeat(x, 'b 1 d -> b t d', t=self.pred_len)
        x = x.repeat([self.pred_len], axis = 1)
        b, t, d = x.shape
        # powers = torch.arange(self.pred_len).to(self._damping_factor.device) + 1
        powers =  ops.arange(self.pred_len) + 1
        powers = powers.view(self.pred_len, 1)
        damping_factors = self.damping_factor ** powers
        # damping_factors = damping_factors.cumsum(dim=0)
        damping_factors = damping_factors.cumsum(axis=0)
        x = x.view(b, t, self.nhead, -1)
        # x = self.dropout(x) * damping_factors.unsqueeze(-1)
        x = self.dropout(x) * ops.ExpandDims()(damping_factors, -1)
        return x.view(b, t, d)
    @property
    def damping_factor(self):
        return ops.sigmoid(self._damping_factor)