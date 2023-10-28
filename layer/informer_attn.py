import sys
sys.path.append("..")
import mindspore
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, XavierNormal, XavierUniform, \
    HeUniform, Uniform, _calculate_fan_in_and_fan_out
from mindspore.nn.cell import Cell
from .basic import _Linear, Dropout
import numpy as np
from math import sqrt


def linear(x, w, b):
    """inner linear"""
    out = ops.matmul(x, w.swapaxes(-1, -2))
    if b is not None:
        out = out + b
    return out

def _in_projection(q, k, v, w_q, w_k, w_v, b_q=None, b_k=None, b_v=None):
    """in projection function"""
    Eq, Ek, Ev = q.shape[-1], k.shape[-1], v.shape[-1]
    w_q_shape, w_k_shape, w_v_shape = w_q.shape, w_k.shape, w_v.shape
    if w_q_shape != (Eq, Eq):
        raise ValueError(f"Expecting query weights shape of {(Eq, Eq)}, but got {w_q_shape}")
    if w_k_shape != (Eq, Ek):
        raise ValueError(f"Expecting key weights shape of {(Eq, Ek)}, but got {w_k_shape}")
    if w_v_shape != (Eq, Ev):
        raise ValueError(f"Expecting value weights shape of {(Eq, Ev)}, but got {w_v_shape}")
    if b_q is not None:
        b_q_shape = b_q.shape
        if b_q_shape != (Eq,):
            raise ValueError(f"Expecting query bias shape of {(Eq,)}, but got {b_q_shape}")
    if b_k is not None:
        b_k_shape = b_k.shape
        if b_k_shape != (Eq,):
            raise ValueError(f"Expecting key bias shape of {(Eq,)}, but got {b_k_shape}")
    if b_v is not None:
        b_v_shape = b_v.shape
        if b_v_shape != (Eq,):
            raise ValueError(f"Expecting value bias shape of {(Eq,)}, but got {b_v_shape}")
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

def _in_projection_packed(q, k, v, w, b, k_is_v, q_is_k):
    """in projecktion packed function"""
    E = q.shape[-1]
    if k_is_v:
        if q_is_k:
            # self-attention
            return linear(q, w, b).tensor_split(3, axis=-1)
        # encoder-decoder attention
        w_q, w_kv = w.split([E, E * 2])
        if b is None:
            b_q = b_kv = None
        else:
            b_q, b_kv = b.split([E, E * 2])
        return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).tensor_split(2, axis=-1)
    w_q, w_k, w_v = w.tensor_split(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.tensor_split(3)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

class ProbAttention(Cell):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 embed_dim=None, num_heads=None, dropout=0., add_zero_attn=False, batch_first=False, kdim=None, vdim=None):
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
        self.in_proj_bias = Parameter(initializer('zeros', (3 * embed_dim)), 'in_proj_bias')

        # self.out_proj = _Linear(embed_dim, embed_dim, has_bias=True)

        self.add_zero_attn = add_zero_attn

        self.k_is_v = False
        self.q_is_k = False

        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = Dropout(p=attention_dropout)

    def __call__(self, *args, **kwargs):
        query = kwargs.get('query', args[0])
        key = kwargs.get('key', args[1])
        value = kwargs.get('value', args[2])
        self.k_is_v = key is value
        self.q_is_k = query is key
        return super().__call__(*args, **kwargs)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        # K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        K_expand = ops.ExpandDims()(K, 2).broadcast_to((B, H, L_Q, L_K, E))

        # real U = U_part(factor*ln(L_k))*L_q

        # index_sample = torch.randint(L_K, (L_Q, sample_k))
        index_sample = ops.randint(0, L_K, (L_Q, sample_k))   # can not work with unknown reason
        # index_sample = mindspore.Tensor(np.random.randint(low=0, high=L_K, size=(L_Q, sample_k))).astype(mindspore.int32)

        # K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        K_sample = K_expand[:, :,  ops.ExpandDims()(ops.arange(L_Q), 1), index_sample, :]
        # Q_K_sample = torch.matmul(
        #     Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
        Q_K_sample = ops.matmul(
            ops.ExpandDims()(Q, -2), K_sample.transpose(0, 1, 2, 4, 3)).squeeze()

        # find the Top_k query with sparisty measurement
        # M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M = Q_K_sample.max(axis = -1)[0] - ops.div(Q_K_sample.sum(-1), L_K)

        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[ops.arange(B)[:, None, None],
                   ops.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        # Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k
        Q_K = ops.matmul(Q_reduce, K.transpose(0, 1, 3, 2))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = ops.mean(V, axis=-2)

            # contex = V_sum.unsqueeze(-2).expand(B, H,
            #                                     L_Q, V_sum.shape[-1]).clone()
            contex = ops.ExpandDims()(V_sum, -2).broadcast_to((B, H, L_Q, V_sum.shape[-1]))
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            # contex = V.cumsum(dim=-2)
            contex = V.cumsum(axis=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores)
            # scores.masked_fill_(attn_mask.mask, -np.inf)
            scores = ops.masked_fill(scores, attn_mask.mask, -np.inf)
        # attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
        attn = ops.softmax(scores, axis=-1)  # nn.Softmax(dim=-1)(scores)

        # context_in[ops.arange(B)[:, None, None], ops.arange(H)[None, :, None], index, :] = ops.matmul(attn, V).type_as(context_in)
        context_in[ops.arange(B)[:, None, None], ops.arange(H)[None, :, None], index, :] = ops.matmul(attn, V)

        if self.output_attention:
            # attns = (torch.ones([B, H, L_V, L_V]) /
            #          L_V).type_as(attn).to(attn.device)
            ones = ops.Ones()
            attns = ones((B, H, L_V, L_V), mindspore.float32) / L_V
            attns[ops.arange(B)[:, None, None], ops.arange(H)[
                                                  None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def construct(self, query: Tensor, key: Tensor, value: Tensor, attn_mask, tau=None, delta=None):
        B, L, _ = query.shape
        _, S, _ = key.shape
        H = self.num_heads
        if not self._qkv_same_embed_dim:
            b_q, b_k, b_v = self.in_proj_bias.tensor_split(3)
            q, k, v = _in_projection(query, key, value, self.q_proj_weight, self.k_proj_weight, self.v_proj_weight, b_q, b_k, b_v)
        else:
            q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias, self.k_is_v, self.q_is_k)
        q = q.view(B, L, H, -1)
        k = k.view(B, S, H, -1)
        v = v.view(B, S, H, -1)

        B, L_Q, H, D = q.shape
        _, L_K, _, _ = k.shape
        queries = q.transpose(0, 2, 1, 3)
        keys = k.transpose(0, 2, 1, 3)
        values = v.transpose(0, 2, 1, 3)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K

        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)
        # return context.contiguous(), attn
        return context, attn


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        # _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        ones = ops.Ones()
        _mask = ones((L, scores.shape[-1]), mindspore.numpy.bool_).triu(1)
        # _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        _mask_ex = _mask[None, None, :].broadcast_to((B, H, L, scores.shape[-1]))
        indicator = _mask_ex[ops.arange(B)[:, None, None],
                    ops.arange(H)[None, :, None],
                    index, :]
        self._mask = indicator.view(scores.shape)
    @property
    def mask(self):
        return self._mask