import sys
sys.path.append("..")
import copy
import math
from typing import Union, Optional
import mindspore
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, XavierNormal, XavierUniform, \
    HeUniform, Uniform, _calculate_fan_in_and_fan_out
from mindspore.nn.cell import Cell
from layer.basic import _Linear, Dropout
from layer.activation import ReLU, GELU
from layer.normalization import LayerNorm
from layer.container import CellList
from layer.Embed import DataEmbedding
from layer.basic import _Linear
from layer.pyraformer_attn import EncoderLayer
from layer.reformer_attn import LSHSelfAttention

class Encoder(Cell):
    """ A encoder model with self attention mechanism. """

    def __init__(self, args, window_size, inner_size):
        super().__init__()
        d_bottleneck = args.d_model//4

        self.mask, self.all_size = get_mask(
            args.seq_len, window_size, inner_size)
        self.indexes = refer_points(self.all_size, window_size)
        # self.layers = nn.ModuleList([
        #     EncoderLayer(configs.d_model, configs.d_ff, configs.n_heads, dropout=configs.dropout,
        #                  normalize_before=False) for _ in range(configs.e_layers)
        # ])  # naive pyramid attention
        self.layers = mindspore.nn.CellList([
            EncoderLayer(args.d_model, args.d_ff, args.n_heads, dropout=args.dropout,
                         normalize_before=False, batch_first = True) for _ in range(args.e_layers)
        ])
        self.enc_embedding = DataEmbedding(
            args.enc_in, args.d_model, args.dropout)
        self.conv_layers = Bottleneck_Construct(
            args.d_model, window_size, d_bottleneck)

    def construct(self, x_enc, x_mark_enc):
        seq_enc = self.enc_embedding(x_enc, x_mark_enc)

        # mask = self.mask.repeat(len(seq_enc), 1, 1)
        mask = mindspore.numpy.tile(self.mask, (len(seq_enc), 1, 1))
        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc = self.layers[i](seq_enc, mask)
        # indexes = self.indexes.repeat(seq_enc.shape[0], 1, 1, seq_enc.shape[2])
        indexes = mindspore.numpy.tile(self.indexes, (seq_enc.shape[0], 1, 1, seq_enc.shape[2]))
        indexes = indexes.view(seq_enc.shape[0], -1, seq_enc.shape[2])
        # all_enc = torch.gather(seq_enc, 1, indexes)
        all_enc = ops.gather_elements(seq_enc, 1, indexes)
        seq_enc = all_enc.view(seq_enc.shape[0], self.all_size[0], -1)

        return seq_enc



class Pyraformer(Cell):
    """
    Pyraformer: Pyramidal attention to reduce complexity
    Paper link: https://openreview.net/pdf?id=0EXmFzUn5I
    """

    def __init__(self, args, window_size=[4, 4], inner_size=5):
        """
        window_size: list, the downsample window size in pyramidal attention.
        inner_size: int, the size of neighbour attention
        """
        super().__init__()
        self.task_name = args.task_name
        self.pred_len = args.pred_len
        self.d_model = args.d_model

        if self.task_name == 'short_term_forecast':
            window_size = [2, 2]
        self.encoder = Encoder(args, window_size, inner_size)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = _Linear(
                (len(window_size) + 1) * self.d_model, self.pred_len * args.enc_in)


    def long_forecast(self, src, src_mark, tgt, tgt_mark, mask=None):
        enc_out = self.encoder(src, src_mark)[:, -1, :]
        dec_out = self.projection(enc_out).view(
            enc_out.shape[0], self.pred_len, -1)
        return dec_out



    def construct(self, src, src_mark, tgt, tgt_mark, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(src, src_mark, tgt, tgt_mark)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None


def get_mask(input_size, window_size, inner_size):
    """Get the attention mask of PAM-Naive"""
    # Get the size of all layers
    all_size = []
    all_size.append(input_size)
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)

    seq_length = sum(all_size)
    # mask = torch.zeros(seq_length, seq_length)
    mask = ops.zeros((seq_length, seq_length))

    # get intra-scale mask
    inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = max(i - inner_window, start)
            right_side = min(i + inner_window + 1, start + all_size[layer_idx])
            mask[i, left_side:right_side] = 1

    # get inter-scale mask
    for layer_idx in range(1, len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = (start - all_size[layer_idx - 1]) + \
                (i - start) * window_size[layer_idx - 1]
            if i == (start + all_size[layer_idx] - 1):
                right_side = start
            else:
                right_side = (
                    start - all_size[layer_idx - 1]) + (i - start + 1) * window_size[layer_idx - 1]
            mask[i, left_side:right_side] = 1
            mask[left_side:right_side, i] = 1

    mask = (1 - mask).bool()

    return mask, all_size

def refer_points(all_sizes, window_size):
    """Gather features from PAM's pyramid sequences"""
    input_size = all_sizes[0]
    # indexes = torch.zeros(input_size, len(all_sizes))
    indexes = ops.zeros((input_size, len(all_sizes)))

    for i in range(input_size):
        indexes[i][0] = i
        former_index = i
        for j in range(1, len(all_sizes)):
            start = sum(all_sizes[:j])
            inner_layer_idx = former_index - (start - all_sizes[j - 1])
            former_index = start + \
                min(inner_layer_idx // window_size[j - 1], all_sizes[j] - 1)
            indexes[i][j] = former_index

    # indexes = indexes.unsqueeze(0).unsqueeze(3)
    indexes = ops.ExpandDims()(indexes, 0)
    indexes = ops.ExpandDims()(indexes, 3)

    return indexes.long()

class Bottleneck_Construct(Cell):
    """Bottleneck convolution CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super(Bottleneck_Construct, self).__init__()
        if not isinstance(window_size, list):
            # self.conv_layers = nn.ModuleList([
            #     ConvLayer(d_inner, window_size),
            #     ConvLayer(d_inner, window_size),
            #     ConvLayer(d_inner, window_size)
            # ])
            self.conv_layers = mindspore.nn.CellList([
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size)
            ])
        else:
            self.conv_layers = []
            for i in range(len(window_size)):
                self.conv_layers.append(ConvLayer(d_inner, window_size[i]))
            # self.conv_layers = nn.ModuleList(self.conv_layers)
            self.conv_layers = mindspore.nn.CellList(self.conv_layers)
        self.up = _Linear(d_inner, d_model)
        self.down = _Linear(d_model, d_inner)
        # self.norm = nn.LayerNorm(d_model)
        self.norm = mindspore.nn.LayerNorm((d_model,))

    def construct(self, enc_input):
        temp_input = self.down(enc_input).permute(0, 2, 1)
        all_inputs = []
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)
            all_inputs.append(temp_input)

        # all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = ops.concat(all_inputs, 2).transpose(0, 2, 1)
        all_inputs = self.up(all_inputs)
        # all_inputs = torch.cat([enc_input, all_inputs], dim=1)
        all_inputs = ops.concat([enc_input, all_inputs], 1)

        all_inputs = self.norm(all_inputs)
        return all_inputs

class ConvLayer(Cell):
    def __init__(self, c_in, window_size):
        super(ConvLayer, self).__init__()
        # self.downConv = mindspore.nn.Conv1d(in_channels=c_in,
        #                           out_channels=c_in,
        #                           kernel_size=window_size,
        #                           stride=window_size)
        # self.norm = nn.BatchNorm1d(c_in)
        # self.activation = nn.ELU()
        self.downConv = mindspore.nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=window_size, stride=window_size, has_bias = True)
        self.norm = mindspore.nn.BatchNorm1d(c_in)
        self.activation = mindspore.nn.ELU()
    def construct(self, x):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x