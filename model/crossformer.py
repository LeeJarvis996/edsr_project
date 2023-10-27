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
from layer.Embed import DataEmbedding_wo_pos
from layer.basic import _Linear
from layer.Embed import PatchEmbedding
from math import ceil
from layer.crossformer_attn import scale_block


class Encoder(Cell):
    def __init__(self, attn_layers):
        super(Encoder, self).__init__()
        # self.encode_blocks = nn.ModuleList(attn_layers)
        self.encode_blocks = CellList(attn_layers)
    def forward(self, x):
        encode_x = []
        encode_x.append(x)
        for block in self.encode_blocks:
            x, attns = block(x)
            encode_x.append(x)
        return encode_x, None




class Model(Cell):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """
    ### enc_in == d_feat
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.d_feat
        self.seq_len = configs.seq_len
        # self.pred_len = configs.pred_len
        self.pred_len = self.seq_len
        self.seg_len = 12
        self.win_size = 2

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * self.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (configs.e_layers - 1)))
        self.head_nf = configs.d_feat * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(configs.d_feat, self.seg_len, self.seg_len, self.pad_in_len - configs.seq_len, 0)
        # self.enc_pos_embedding = nn.Parameter(
        #     torch.randn(1, configs.d_feat, self.in_seg_num, configs.d_feat))
        self.enc_pos_embedding = Parameter(mindspore.numpy.randn((1, configs.d_feat, self.in_seg_num, configs.d_feat)), requires_grad=True)

        # self.pre_norm = nn.LayerNorm(configs.d_feat)
        self.pre_norm = mindspore.nn.LayerNorm((configs.d_feat,))


        # Encoder
        self.encoder = Encoder(
            [
                scale_block(configs, 1 if l is 0 else self.win_size, configs.d_feat, configs.n_heads, configs.d_ff,
                            1, configs.dropout,
                            self.in_seg_num if l is 0 else ceil(self.in_seg_num / self.win_size ** l), configs.factor
                            ) for l in range(configs.e_layers)
            ]
        )
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(self.head_nf * configs.d_feat, 1)

    def regression(self, x_enc, x_mark_enc):
        # embedding
        x_enc = x_enc.reshape(len(x_enc), self.enc_in, -1)  # [N, F, T]
        x_enc = x_enc.permute(0, 2, 1)  # [N, T, F]
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)
        # Output from Non-stationary Transformer
        output = self.flatten(enc_out[-1].permute(0, 1, 3, 2))
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.regression(x_enc, x_mark_enc)
        return dec_out.squeeze()  # [B, 1]