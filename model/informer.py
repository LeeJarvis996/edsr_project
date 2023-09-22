import sys
sys.path.append("..")
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
from layer.basic import _Linear
from layer.informer_attn import ProbAttention
from layer.reformer_attn import LSHSelfAttention
import time
import copy



class ConvLayer(Cell):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        # self.downConv = nn.Conv1d(in_channels=c_in,
        #                           out_channels=c_in,
        #                           kernel_size=3,
        #                           padding=2,
        #                           padding_mode='circular')
        # self.norm = nn.BatchNorm1d(c_in)
        # self.activation = nn.ELU()
        # self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.downConv = mindspore.nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=2, pad_mode = 'pad',
                                            has_bias=True)
        self.norm = mindspore.nn.BatchNorm1d(c_in)
        self.activation = mindspore.nn.ELU()
        self.maxPool = mindspore.nn.MaxPool1d(kernel_size=3, stride=2, padding=1, pad_mode='pad')
    def construct(self, x):
        x = self.downConv(x.transpose(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(0, 2, 1)
        return x

class InformerDecoderLayer(Cell):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Cell, callable] = 'relu', layer_norm_eps: float = 1e-5,
                 batch_first: bool = False, norm_first: bool = False, args = None, d_values = None):
        super().__init__()
        d_values = d_values or (d_model // nhead)
        self.self_attn = ProbAttention(True, args.factor, attention_dropout=args.dropout, output_attention=False, embed_dim=args.d_model,
                                       num_heads = args.n_heads, batch_first=batch_first)
        self.out_projection1 = _Linear(d_values * nhead, d_model)

        self.multihead_attn = ProbAttention(False, args.factor, attention_dropout=args.dropout, output_attention=False, embed_dim=args.d_model,
                                       num_heads = args.n_heads, batch_first=batch_first)
        self.out_projection2 = _Linear(d_values * nhead, d_model)

        # feedforward layer
        self.conv1 = mindspore.nn.Conv1d(in_channels=d_model, out_channels=dim_feedforward, kernel_size=1, has_bias=True)
        self.conv2 = mindspore.nn.Conv1d(in_channels=dim_feedforward, out_channels=d_model, kernel_size=1, has_bias=True)
        self.norm1 = LayerNorm((d_model,), epsilon=layer_norm_eps)
        self.norm2 = LayerNorm((d_model,), epsilon=layer_norm_eps)
        self.norm3 = LayerNorm((d_model,), epsilon=layer_norm_eps)
        self.dropout = Dropout(p=dropout)
        self.dropout1 = Dropout(p=dropout)
        self.dropout2 = Dropout(p=dropout)
        self.dropout3 = Dropout(p=dropout)
        self.norm_first = norm_first

        if not isinstance(activation, str) and not isinstance(activation, Cell) \
            and not callable(activation):
            raise ValueError(f"The argument 'activation' must be str, callable or Cell instance,"
                             f" but get {activation}.")
        if isinstance(activation, Cell) and (not isinstance(activation, ReLU) or \
                                             not isinstance(activation, GELU)):
            raise ValueError(f"The argument 'activation' must be nn.ReLU or nn.GELU instance,"
                             f" but get {activation}.")
        if callable(activation) and (activation is not ops.relu or \
                                     activation is not ops.gelu):
            raise ValueError(f"The argument 'activation' must be ops.relu or ops.gelu instance,"
                             f" but get {activation}.")
        # string inputs of activation
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def construct(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                  memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                  memory_key_padding_mask: Optional[Tensor] = None, is_causal:bool = False):
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, is_causal)
            y = x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            # x = x + self._ff_block(self.norm3(x))
            y = self.dropout3(self.activation(self.conv1(y.transpose(0, 2, 1))))
            y = self.dropout3(self.conv2(y).transpose(0, 2, 1))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, is_causal))
            y = x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            y = self.dropout3(self.activation(self.conv1(y.transpose(0, 2, 1))))
            y = self.dropout3(self.conv2(y).transpose(0, 2, 1))

        #     x = self.norm3(x + self._ff_block(x))
        # return x
        return self.norm3(x + y)

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        B, L, _ = x.shape
        _, S, _ = x.shape
        x = self.self_attn(x, x, x,
                           attn_mask)[0].view(B, L, -1)
        x = self.out_projection1(x)
        return self.dropout1(x)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        B, L, _ = x.shape
        _, S, _ = x.shape
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask)[0].view(B, L, -1)
        x = self.out_projection2(x)
        return self.dropout2(x)




class InformerDecoder(Cell):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None, projection=None):
        super(InformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.projection = projection

    def construct(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                  memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                  memory_key_padding_mask: Optional[Tensor] = None, is_causal:bool = False):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask, is_causal=is_causal)

        if self.norm is not None:
            output = self.norm(output)
        if self.projection is not None:
            output = self.projection(output)

        return output

class InformerEncoderLayer(Cell):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Cell, callable] = 'relu', layer_norm_eps: float = 1e-5,
                 batch_first: bool = False, norm_first: bool = False, Attn_func = None, d_values=None):
        super().__init__()

        d_values = d_values or (d_model // nhead)
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.self_attn = Attn_func
        self.out_projection = _Linear(d_values * nhead, d_model)
        # feedforward layer
        self.conv1 = mindspore.nn.Conv1d(in_channels=d_model, out_channels=dim_feedforward, kernel_size=1,
                                         has_bias=True)
        self.conv2 = mindspore.nn.Conv1d(in_channels=dim_feedforward, out_channels=d_model, kernel_size=1,
                                         has_bias=True)
        self.norm1 = LayerNorm((d_model,), epsilon=layer_norm_eps)
        self.norm2 = LayerNorm((d_model,), epsilon=layer_norm_eps)
        self.dropout = Dropout(p=dropout)
        self.dropout1 = Dropout(p=dropout)
        self.dropout2 = Dropout(p=dropout)
        self.dropout3 = Dropout(p=dropout)
        # self.linear1 = _Linear(d_model, dim_feedforward)
        # self.linear2 = _Linear(dim_feedforward, d_model)
        self.norm_first = norm_first

        if not isinstance(activation, str) and not isinstance(activation, Cell) \
            and not callable(activation):
            raise ValueError(f"The argument 'activation' must be str, callable or Cell instance,"
                             f" but get {activation}.")
        if isinstance(activation, Cell) and (not isinstance(activation, ReLU) or \
                                             not isinstance(activation, GELU)):
            raise ValueError(f"The argument 'activation' must be nn.ReLU or nn.GELU instance,"
                             f" but get {activation}.")
        if callable(activation) and (activation is not ops.relu or \
                                     activation is not ops.gelu):
            raise ValueError(f"The argument 'activation' must be ops.relu or ops.gelu instance,"
                             f" but get {activation}.")
        # string inputs of activation
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def construct(self, src: Tensor, src_mask: Optional[Tensor] = None,
                  src_key_padding_mask: Optional[Tensor] = None, is_causal:bool = False):
        # print("src_key_padding_mask",src_key_padding_mask)
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != mindspore.bool_ and not ops.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")

        x = src
        if self.norm_first:
            y = x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
            y = self.dropout3(self.activation(self.conv1(y.transpose(0,2,1))))
            y = self.dropout3(self.conv2(y).transpose(0,2,1))
            # x = x + self._ff_block(self.norm2(x))
        else:
            y = x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal))
            y = self.dropout3(self.activation(self.conv1(y.transpose(0,2,1))))
            y = self.dropout3(self.conv2(y).transpose(0,2,1))
            # x = self.norm2(x + self._ff_block(x))
        # return x
        return self.norm2(x + y)

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        B, L, _ = x.shape
        _, S, _ = x.shape
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask
        )[0].view(B, L, -1)
        x = self.out_projection(x)
        return self.dropout1(x)



class InformerEncoder(Cell):
    __constants__ = ['norm']
    # def __init__(self, encoder_layer, num_layers, norm=None):
    def __init__(self, attn_layers, conv_layers, norm=None):
        super(InformerEncoder, self).__init__()
        # self.layers = _get_clones(encoder_layer, num_layers)
        # self.num_layers = num_layers
        self.attn_layers = mindspore.nn.CellList(attn_layers)
        self.conv_layers = mindspore.nn.CellList(conv_layers) if conv_layers is not None else None
        self.norm = norm

    def construct(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, is_causal: bool = False):
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != mindspore.bool_ and not ops.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        output = src
        src_key_padding_mask_for_layers = src_key_padding_mask
        # for i, mod in enumerate(self.layers[:-1]):
        #     output = mod(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask_for_layers, is_causal = is_causal)
        #     output = ConvLayer(output)
        # output = self.layers[-1](output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask_for_layers,
        #              is_causal=is_causal)

        for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
            output = attn_layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask_for_layers, is_causal = is_causal)
            output = conv_layer(output)
        output = self.attn_layers[-1](output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask_for_layers, is_causal = is_causal)
        if self.norm is not None:
            output = self.norm(output)
        return output


class Informer(Cell):
    """
    Reformer with O(LlogL) complexity
    Paper link: https://openreview.net/forum?id=rkgNKkHtvB
    """
    def __init__(self, args, layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False):
        super(Informer, self).__init__()

        self.task_name = args.task_name
        self.pred_len = args.pred_len
        self.seq_len = args.seq_len

        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq, args.dropout)

        # Encoder
        encoder_layer = [InformerEncoderLayer(args.d_model, args.n_heads, args.d_ff, args.dropout,
                                                args.activation, layer_norm_eps, batch_first, norm_first,
                                                Attn_func=ProbAttention(False, args.factor, attention_dropout=args.dropout,
                                                    output_attention=args.output_attention, embed_dim = args.d_model, num_heads = args.n_heads,
                                                                        batch_first=batch_first))
                         for l in range(args.e_layers)]
        conv_layer = [
                    ConvLayer(args.d_model) for l in range(args.e_layers - 1)] if args.distil and ('forecast' in args.task_name) else None
        encoder_norm = LayerNorm((args.d_model,), epsilon=layer_norm_eps)
        # self.encoder = InformerEncoder(encoder_layer, args.e_layers, encoder_norm)
        self.encoder = InformerEncoder(encoder_layer, conv_layer, encoder_norm)

        self.dec_embedding = DataEmbedding(args.dec_in, args.d_model, args.embed, args.freq, args.dropout)

        decoder_layer = InformerDecoderLayer(args.d_model, args.n_heads, args.d_ff, args.dropout,
                                                args.activation, layer_norm_eps, batch_first, norm_first, args=args)
        decoder_norm = LayerNorm((args.d_model,), epsilon=layer_norm_eps)
        projection = _Linear(args.d_model, args.c_out, has_bias=True)
        self.decoder = InformerDecoder(decoder_layer, args.d_layers, norm=decoder_norm, projection=projection)

        for _, p in self.parameters_and_names():
            if p.ndim > 1:
                p.set_data(initializer('xavier_uniform', p.shape, p.dtype))

        self.d_model = args.d_model
        self.nhead = args.n_heads

        self.batch_first = batch_first

    def construct(self, src: Tensor, src_mark: Tensor, tgt: Tensor, tgt_mark: Tensor, src_mask: Optional[Tensor] = None,
                  tgt_mask: Optional[Tensor] = None,
                  memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                  tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None):
        is_batched = src.ndim == 3
        if self.batch_first:
            src_batch_size = src.shape[0]
            tgt_batch_size = src.shape[0]
        else:
            src_batch_size = src.shape[1]
            tgt_batch_size = src.shape[1]
        if src_batch_size != tgt_batch_size and is_batched:
            raise ValueError("The number of batch size for 'src' and 'tgt' must be equal.")
        src = self.enc_embedding(src, src_mark)
        tgt = self.dec_embedding(tgt, tgt_mark)
        # print("encoder")
        memory = self.encoder(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=False)

        # # print("decoder")
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask, is_causal=True,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output


def _get_activation_fn(activation: str):
    if activation == "relu":
        return ops.relu
    if activation == "gelu":
        return ops.gelu

    raise ValueError(f"The activation must be relu/gelu, but get {activation}")

def _get_clones(module, N):
    return CellList([copy.deepcopy(module) for i in range(N)])