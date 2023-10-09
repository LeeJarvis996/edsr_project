from typing import Union, Optional
import mindspore
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, XavierNormal, XavierUniform, \
    HeUniform, Uniform, _calculate_fan_in_and_fan_out
from mindspore.ops.function.nn_func import _check_qkv_shape, _check_kpm_shape, _in_projection_packed, _in_projection, _inner_pad
from mindspore.nn.cell import Cell
from .basic import _Linear, Dropout
import math
import numpy as np


class series_decomp(Cell):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
    def construct(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class moving_avg(Cell):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        # self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        self.avg = mindspore.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0 , pad_mode="pad")

    def construct(self, x):
        # padding on the both ends of time series
        # front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        # end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        # x = torch.cat([front, x, end], dim=1)
        # x = self.avg(x.permute(0, 2, 1))
        # x = x.permute(0, 2, 1)
        front = mindspore.numpy.tile(x[:, 0:1, :], (1, (self.kernel_size - 1) // 2, 1))
        end = mindspore.numpy.tile(x[:, -1:, :], (1, (self.kernel_size - 1) // 2, 1))
        x = ops.concat([front, x, end], 1)
        x = self.avg(x.transpose(0, 2, 1).astype('Float32'))  # error
        x = x.transpose(0, 2, 1)
        return x


class AutoCorrelationLayer(Cell):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = _Linear(d_model, d_keys * n_heads)
        self.key_projection = _Linear(d_model, d_keys * n_heads)
        self.value_projection = _Linear(d_model, d_values * n_heads)
        self.out_projection = _Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def construct(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).astype(mindspore.float32).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn



class AutoCorrelation(Cell):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = Dropout(p=attention_dropout)

    def time_delay_agg_training(self, values, corr):
        # print("values", values)
        # print("corr", corr)
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        # mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        # index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        # weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        top_k = int(self.factor * math.log(length))
        mean_value = ops.mean(ops.mean(corr, axis=1), axis=1)
        index = mindspore.ops.topk(ops.mean(mean_value, axis=0), top_k, dim=-1)[1]
        weights = ops.stack([mean_value[:, index[i]] for i in range(top_k)], axis=-1)
        # update corr
        # tmp_corr = torch.softmax(weights, dim=-1)
        tmp_corr = ops.softmax(weights, axis=-1)
        # aggregation
        tmp_values = values
        # delays_agg = torch.zeros_like(values).float()
        delays_agg = ops.zeros_like(values, dtype=mindspore.float32)
        for i in range(top_k):
            # pattern = torch.roll(tmp_values, -int(index[i]), -1)
            # delays_agg = delays_agg + pattern * \
            #              (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
            '''
            pattern = ops.roll(tmp_values, -int(index[i]), -1)
            Unsupported op [Roll] on CPU -> we use np.roll instead.
            '''
            pattern = Tensor(np.roll(tmp_values.asnumpy(), shift=-int(index[i]),axis=-1))
            a = mindspore.numpy.tile(ops.ExpandDims()(ops.ExpandDims()(ops.ExpandDims()(tmp_corr[:, i],1),1),1), (1, head, channel, length))
            delays_agg = delays_agg + pattern * a
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        # init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        init_index = ops.ExpandDims()(ops.ExpandDims()(ops.ExpandDims()(ops.arange(length).astype(mindspore.float32), 0), 0),0)
        init_index = mindspore.numpy.tile(init_index, (batch, head, channel, 1))
        # find top k
        # mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        # weights, delay = torch.topk(mean_value, top_k, dim=-1)
        top_k = int(self.factor * math.log(length))
        mean_value = ops.mean(ops.mean(corr, axis=1), axis=1)
        weights, delay = ops.topk(mean_value, top_k, dim=-1)
        # update corr
        # tmp_corr = torch.softmax(weights, dim=-1)
        tmp_corr = ops.softmax(weights, axis=-1)
        # aggregation
        # tmp_values = values.repeat(1, 1, 1, 2)
        # delays_agg = torch.zeros_like(values).float()
        tmp_values = mindspore.numpy.tile(values, (1, 1, 1, 2))
        delays_agg = ops.zeros_like(values, dtype=mindspore.float32)
        for i in range(top_k):
            # tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            a = mindspore.numpy.tile(ops.ExpandDims()(ops.ExpandDims()(ops.ExpandDims()(delay[:, i],1),1),1), (1, head, channel, length))
            tmp_delay = init_index + a
            # pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            pattern = ops.gather_elements(tmp_values, dim=-1, index=tmp_delay.astype('Int64'))  # error
            # delays_agg = delays_agg + pattern * \
            #              (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
            temp = ops.ExpandDims()(ops.ExpandDims()(ops.ExpandDims()(tmp_corr[:, i],1),1),1)
            a = mindspore.numpy.tile(temp, (1, head, channel, length))
            delays_agg = delays_agg + pattern * a
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        # init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        init_index = ops.ExpandDims()(ops.ExpandDims()(ops.ExpandDims()(ops.arange(length).astype(mindspore.float32), 0), 0),0)
        init_index = mindspore.numpy.tile(init_index, (batch, head, channel, 1))
        # find top k
        top_k = int(self.factor * math.log(length))
        # weights, delay = torch.topk(corr, top_k, dim=-1)
        weights, delay = ops.topk(corr, top_k, dim=-1)
        # update corr
        # tmp_corr = torch.softmax(weights, dim=-1)
        tmp_corr = ops.softmax(weights, axis=-1)

        # aggregation
        # tmp_values = values.repeat(1, 1, 1, 2)
        # delays_agg = torch.zeros_like(values).float()
        tmp_values = mindspore.numpy.tile(values, (1, 1, 1, 2))
        delays_agg = ops.zeros_like(values, dtype=mindspore.float32)
        for i in range(top_k):
            # tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            tmp_delay = init_index + ops.ExpandDims()(delay[..., i], -1)
            # pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            pattern = ops.gather_elements(tmp_values, dim=-1, index=tmp_delay)
            # delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
            delays_agg = delays_agg + pattern * (ops.ExpandDims()(tmp_corr[..., i], -1))
        return delays_agg


    def construct(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            # zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            zeros = ops.zeros_like(queries[:, :(L - S), :], dtype=mindspore.float32)
            # values = torch.cat([values, zeros], dim=1)
            # keys = torch.cat([keys, zeros], dim=1)
            values = ops.concat([values, zeros], 1)
            keys = ops.concat([keys, zeros], 1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        # period-based dependencies
        # q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        # k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        # res = q_fft * torch.conj(k_fft)
        # corr = torch.fft.irfft(res, dim=-1)
        fft_net = ops.FFTWithSize(inverse = False, real = True, signal_ndim = 1)
        q_fft = fft_net(queries.transpose(0, 2, 3, 1))
        # print("q_fft", queries.transpose(0, 2, 3, 1)[0][0])
        fft_net = ops.FFTWithSize(inverse = False, real = True, signal_ndim = 1)
        k_fft = fft_net(keys.transpose(0, 2, 3, 1))
        '''
        res = q_fft * ops.conj(k_fft)
        TypeError: For 'Mul', gradient not support for complex type currently.
        For now, we must transform mindspore.Tensor -> numpy to do calculation.
        If gradient support is added in the future, please use this code:
        # res = q_fft * ops.conj(k_fft)
        '''
        res = q_fft.asnumpy() * ops.conj(k_fft).asnumpy()
        # print("res", res.shape)
        net_ = ops.FFTWithSize(inverse=True, real=True, signal_ndim=1)
        corr = net_(Tensor(res))
        # print("corr", corr[0])
        # print("res", res[0])
        # print(corr.shape)
        # print(res.shape)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.transpose(0, 2, 3, 1), corr).transpose(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.transpose(0, 2, 3, 1), corr).transpose(0, 3, 1, 2)

        if self.output_attention:
            return (V, corr.transpose(0, 3, 1, 2))
        else:
            return (V, None)