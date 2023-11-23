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

__all__ = ['MultiheadAttention', 'TransformerEncoderLayer', 'TransformerDecoderLayer',
           'TransformerEncoder', 'TransformerDecoder', 'Transformer']



class MultiheadAttention(Cell):
    def __init__(self, embed_dim, num_heads, dropout=0., has_bias=True, add_bias_kv=False,
                 add_zero_attn=False, batch_first=False, kdim=None, vdim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim


        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("The init argument 'embed_dim' must be divisible by 'num_heads'.")

        # if not self._qkv_same_embed_dim:
        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, embed_dim)), 'q_proj_weight')
            self.k_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, self.kdim)), 'k_proj_weight')
            self.v_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, self.vdim)), 'v_proj_weight')
            self.in_proj_weight = None
        else:
            self.in_proj_weight = Parameter(initializer(XavierUniform(), (3 * embed_dim, embed_dim)), 'in_proj_weight')
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None

        if has_bias:
            self.in_proj_bias = Parameter(initializer('zeros', (3 * embed_dim)), 'in_proj_bias')

        else:
            self.in_proj_bias = None
        self.out_proj = _Linear(embed_dim, embed_dim, has_bias=has_bias)

        if add_bias_kv:
            self.bias_k = Parameter(initializer(XavierNormal(), (1, 1, embed_dim)), 'bias_k')
            self.bias_v = Parameter(initializer(XavierNormal(), (1, 1, embed_dim)), 'bias_v')
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.k_is_v = False
        self.q_is_k = False

    def __call__(self, *args, **kwargs):
        query = kwargs.get('query', args[0])
        key = kwargs.get('key', args[1])
        value = kwargs.get('value', args[2])
        self.k_is_v = key is value
        self.q_is_k = query is key
        return super().__call__(*args, **kwargs)

    def construct(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None,
                  key_padding_mask: Optional[Tensor] = None, need_weights: bool = True,  average_attn_weights: bool = True,
                  is_causal: bool = False):
        is_batched = query.ndim == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != mindspore.bool_ and not ops.is_floating_point(key_padding_mask):
                raise ValueError(
                    "only bool and floating types of key_padding_mask are supported")

        if self.batch_first and is_batched:
            # k_is_v and q_is_k preprocess in __call__ since Graph mode do not support `is`
            if self.k_is_v:
                if self.q_is_k:
                    query = key = value = query.swapaxes(1, 0)
                else:
                    query, key = [x.swapaxes(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.swapaxes(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights,
                k_is_v=self.k_is_v, q_is_k=self.q_is_k, is_causal=is_causal)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights,
                k_is_v=self.k_is_v, q_is_k=self.q_is_k, is_causal=is_causal)

        if self.batch_first and is_batched:
            attn_output = attn_output.swapaxes(1, 0)
        # print("need_weights",need_weights)      # False
        if need_weights:
            return attn_output, attn_output_weights
        return (attn_output,)


class TransformerEncoderLayer(Cell):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Cell, callable] = 'relu', layer_norm_eps: float = 1e-5,
                 batch_first: bool = False, norm_first: bool = False, Attn_func = None):
        super().__init__()

        self.self_attn = Attn_func
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
        # return x
        return self.norm2(x + y)

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)



class TransformerDecoderLayer(Cell):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Cell, callable] = 'relu', layer_norm_eps: float = 1e-5,
                 batch_first: bool = False, norm_first: bool = False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
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
            y = self.dropout3(self.activation(self.conv1(y.transpose(0, 2, 1))))
            y = self.dropout3(self.conv2(y).transpose(0, 2, 1))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, is_causal))
            y = x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))

            y = self.dropout3(self.activation(self.conv1(y.transpose(0, 2, 1))))
            y = self.dropout3(self.conv2(y).transpose(0, 2, 1))

        return self.norm3(x + y)

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)



class TransformerEncoder(Cell):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def construct(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, is_causal: bool = False):
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != mindspore.bool_ and not ops.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        output = src
        src_key_padding_mask_for_layers = src_key_padding_mask
        for mod in self.layers:
            output = mod(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask_for_layers, is_causal = is_causal)
        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Cell):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None, projection=None):
        super(TransformerDecoder, self).__init__()
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


class Transformer(Cell):
    def __init__(self, args = None, custom_encoder: Optional[Cell] = None,
                 custom_decoder: Optional[Cell] = None, layer_norm_eps: float = 1e-5,
                 batch_first: bool = True, norm_first: bool = False):
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq, args.dropout)
            encoder_layer = TransformerEncoderLayer(args.d_model, args.n_heads, args.d_ff, args.dropout,
                                                    args.activation, layer_norm_eps, batch_first, norm_first,
                                                    Attn_func=MultiheadAttention(args.d_model, args.n_heads, dropout=args.dropout, batch_first=batch_first))
            encoder_norm = LayerNorm((args.d_model,), epsilon=layer_norm_eps)
            self.encoder = TransformerEncoder(encoder_layer, args.e_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            self.dec_embedding = DataEmbedding(args.dec_in, args.d_model, args.embed, args.freq, args.dropout)
            decoder_layer = TransformerDecoderLayer(args.d_model, args.n_heads, args.d_ff, args.dropout,
                                                    args.activation, layer_norm_eps, batch_first, norm_first)
            decoder_norm = LayerNorm((args.d_model,), epsilon=layer_norm_eps)
            projection = _Linear(args.d_model, args.c_out, has_bias=True)
            self.decoder = TransformerDecoder(decoder_layer, args.d_layers, norm=decoder_norm, projection=projection)

        for _, p in self.parameters_and_names():
            if p.ndim > 1:
                p.set_data(initializer('xavier_uniform', p.shape, p.dtype))

        self.d_model = args.d_model
        self.nhead = args.n_heads
        self.batch_first = batch_first
        self.pred_len = args.pred_len

    def construct(self, src: Tensor, src_mark: Tensor, tgt: Tensor, tgt_mark: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
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
        # print("enc_embedding")
        src = self.enc_embedding(src, src_mark)
        # print("encoder")
        memory = self.encoder(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal = False)
        # print("dec_embedding")
        tgt = self.dec_embedding(tgt, tgt_mark)
        # print("decoder")
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask, is_causal = True,
                              memory_key_padding_mask=memory_key_padding_mask)

        return output[:, -self.pred_len:, :]


def _get_activation_fn(activation: str):
    if activation == "relu":
        return ops.relu
    if activation == "gelu":
        return ops.gelu
    raise ValueError(f"The activation must be relu/gelu, but get {activation}")


def _get_clones(module, N):
    return CellList([copy.deepcopy(module) for i in range(N)])
