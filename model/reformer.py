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
from mindspore.ops.function.nn_func import multi_head_attention_forward
from mindspore.nn.cell import Cell
from layer.basic import _Linear, Dropout
from layer.activation import ReLU, GELU
from layer.normalization import LayerNorm
from layer.container import CellList
from layer.Embed import DataEmbedding
from layer.transformer import TransformerEncoder, TransformerEncoderLayer
from layer.basic import _Linear
from layer.reformer_attn import LSHSelfAttention
import time

class ReformerLayer(Cell):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4, dropout=None):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            # return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)
            return ops.concat([queries, ops.Zeros([B, fill_len, C])], 1)

    # def construct(self, queries, keys, values, attn_mask, tau, delta, ttn_mask,
    #                        key_padding_mask, need_weights, is_causal):
    def construct(self, queries, keys, values, key_padding_mask, need_weights, is_causal, attn_mask):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class Reformer(Cell):
    """
    Reformer with O(LlogL) complexity
    Paper link: https://openreview.net/forum?id=rkgNKkHtvB
    """
    def __init__(self, bucket_size:int = 4, n_hashes:int =4, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Cell, callable] = 'relu', custom_encoder: Optional[Cell] = None,
                 custom_decoder: Optional[Cell] = None, layer_norm_eps: float = 1e-5,
                 batch_first: bool = False, norm_first: bool = False, enc_in: int = 7, embed: str = 'timeF',
                 freq: str = 'h', dec_in: int = 7, c_out: int = 7, task_name: str = 'long_term_forecast', pred_len:int = 96,
                 seq_len:int = 96):
        super(Reformer, self).__init__()

        self.task_name = task_name
        self.pred_len = pred_len
        self.seq_len = seq_len

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        # Encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first,
                                                Attn_func=ReformerLayer(None, d_model, nhead, bucket_size=bucket_size, n_hashes=n_hashes, dropout=dropout))
        encoder_norm = LayerNorm((d_model,), epsilon=layer_norm_eps)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.projection = _Linear(d_model, c_out, has_bias=True)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        x_enc = ops.concat([x_enc, x_dec[:, -self.pred_len:, :]], 1)

        if x_mark_enc is not None:
            # x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
            x_mark_enc = ops.concat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], 1)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        # enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = self.encoder(enc_out, src_mask=x_mark_enc, is_causal=False)
        dec_out = self.projection(enc_out)

        return dec_out  # [B, L, D]

    def imputation(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        enc_out, attns = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        return enc_out  # [B, L, D]

    def construct(self, src, src_mark, tgt, tgt_mark, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(src, src_mark, tgt, tgt_mark)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None