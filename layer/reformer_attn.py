from .basic import _Linear, Dropout
import mindspore
import numpy as np
import mindspore.ops as ops
from mindspore.nn.cell import Cell
from tqdm import trange
from mindspore import Parameter
from functools import partial, reduce, wraps
from operator import mul
import time

TOKEN_SELF_ATTN_VALUE = -5e4

def cache_method_decorator(cache_attr, cache_namespace, reexecute = False):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, key_namespace=None, fetch=False, set_cache=True, **kwargs):
            namespace_str = str(default(key_namespace, ''))
            _cache = getattr(self, cache_attr)
            _keyname = f'{cache_namespace}:{namespace_str}'

            if fetch:
                val = _cache[_keyname]
                if reexecute:
                    fn(self, *args, **kwargs)
            else:
                val = fn(self, *args, **kwargs)
                if set_cache:
                    setattr(self, cache_attr, {**_cache, **{_keyname: val}})
            return val
        return wrapper
    return inner_fn

def process_inputs_chunk(fn, chunks=1, dim=0):
    def inner_fn(*args, **kwargs):
        keys, values, len_args = kwargs.keys(), kwargs.values(), len(args)
        split = ops.Split(output_num = chunks, axis=dim)
        # chunked_args = list(zip(*map(lambda x: x.chunk(chunks, dim=dim), list(args) + list(values))))
        a = list(map(lambda x: split(x), list(args) + list(values)))
        chunked_args = [(a[0][0],a[1][0])]
        # print("chunked_args", chunked_args)
        all_args = map(lambda x: ((x[:len_args], dict(zip(keys, x[len_args:])))), chunked_args)
        # all_args = list(map(lambda x: (x[:len_args]), chunked_args))
        outputs = [fn(*c_args, **c_kwargs) for c_args, c_kwargs in all_args]
        # return tuple(map(lambda x: torch.cat(x, dim=dim), zip(*outputs)))
        return tuple(map(lambda x: ops.concat(x, dim), zip(*outputs)))
    return inner_fn

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def expand_dim(dim, k, t):
    # t = t.unsqueeze(dim)
    t = ops.ExpandDims()(t, dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    # return t.expand(*expand_shape)
    return ops.expand(t, mindspore.Tensor(np.array(*expand_shape), mindspore.int32))

def max_neg_value(tensor):
    tensor = tensor.asnumpy()
    return np.finfo(tensor.dtype).max
    # return -torch.finfo(tensor.dtype).max

def merge_dims(ind_from, ind_to, tensor):
    '''
       tensor: (32, 0, 192, 64): []
       ind_from: 0
       ind_to: 1
       shape: [32, 0, 192, 64]
       arr_slice: slice(0,2,None)
       shape: [0, 192, 64]
    '''
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    # print("shape", shape)
    # print("tensor", tensor)
    return tensor.reshape(*shape)

def default(val, default_val):
    return default_val if val is None else val

def exists(val):
    return val is not None

def cat(tuple_tensor, dim):
    cat = ops.Concat(dim)
    return cat(tuple_tensor)

def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tensor.reshape(-1, last_dim)
    summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
    return ops.concat(summed_tensors, 0).reshape(orig_size)
    # return torch.cat(summed_tensors, dim=0).reshape(orig_size)

def einsum_func1(dropped_vecs, random_rotations):
    '''
    CPU version of MindSpore implementation of:
    torch.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)
    '''
    # batch_size, time, feature = dropped_vecs.shape
    # _, feature, height, width = random_rotations.shape
    # result = ops.normal((batch_size, time, height, width), 0, 1, seed=5)
    #
    # for i in trange(batch_size, desc="einsum_func1"):
    #     for j in range(time):
    #         for k in range(height):
    #             for l in range(width):
    #                 result[i, j, k, l] = ops.ReduceSum()(dropped_vecs[i, j, :] * random_rotations[i, :, k, l])
    # return result.transpose(0, 2, 1, 3)
    # print("einsum_func1")
    import torch
    # print("11", dropped_vecs)
    # print("22", random_rotations)
    a = torch.einsum('btf,bfhi->bhti', torch.from_numpy(dropped_vecs.asnumpy()), torch.from_numpy(random_rotations.asnumpy()))
    return mindspore.Tensor(a.numpy())

def einsum_func2(x, y):
    ''''bhie,bhje->bhij'''
    # batch_size, height, dim_x, dim_y = x.shape[0], x.shape[1], x.shape[2], y.shape[2]
    # # 创建一个用于存储结果的空张量，形状为 (b, h, i, j)
    # result = ops.zeros((batch_size, height, dim_x, dim_y))
    # for b in trange(batch_size, desc="einsum_func2"):
    #     for h in range(height):
    #         for i in range(dim_x):
    #             for j in range(dim_y):
    #                 # 计算内积并将结果存储在 result 中, result[b, h, i, j] = torch.dot(x[b, h, i], y[b, h, j])
    #                 result[b, h, i, j] = ops.ReduceSum() (x[b, h, i] *  y[b, h, j])
    # return result
    # print("einsum_func2")
    import torch
    a = torch.einsum('bhie,bhje->bhij', torch.from_numpy(x.asnumpy()), torch.from_numpy(y.asnumpy()))
    return mindspore.Tensor(a.numpy())

def einsum_func3(x, y):
    ''''buij,buje->buie'''
    # batch_size, u, i, j = x.shape
    # _, _, _, e = y.shape
    # result = ops.zeros((batch_size, u, i, e))
    # for b in trange(batch_size, desc="einsum_func3"):
    #     for u_idx in range(u):
    #         for i_idx in range(i):
    #             for e_idx in range(e):
    #                 # 计算内积并将结果存储在 result 中
    #                 result[b, u_idx, i_idx, e_idx] = ops.ReduceSum()(x[b, u_idx, i_idx] * y[b, u_idx, :, e_idx])
    # return result
    # print("einsum_func3")
    import torch
    a = torch.einsum('buij,buje->buie', torch.from_numpy(x.asnumpy()), torch.from_numpy(y.asnumpy()))
    return mindspore.Tensor(a.numpy())

def einsum_func4(x, y):
    '''bie,bje->bij'''
    # batch_size, i, e = x.shape
    # _, j, _ = y.shape
    # result = ops.zeros((batch_size, i, j))
    # for b in trange(batch_size, desc="einsum_func4"):
    #     for i_idx in range(i):
    #         for j_idx in range(j):
    #             # 计算内积并将结果存储在 result 中
    #             result[b, i_idx, j_idx] = ops.ReduceSum()(x[b, i_idx] * y[b, j_idx])
    # return result
    # print("einsum_func4")
    import torch
    a = torch.einsum('bie,bje->bij', torch.from_numpy(x.asnumpy()), torch.from_numpy(y.asnumpy()))
    return mindspore.Tensor(a.numpy())

def einsum_func5(x, y):
    '''bij,bje->bie'''
    # batch_size, i, e = x.shape
    # _, j, _ = y.shape
    # result = ops.zeros((batch_size, i, j))
    # for b in range(batch_size):
    #     for i_idx in range(i):
    #         for j_idx in range(j):
    #             # 计算内积并将结果存储在 result 中
    #             result[b, i_idx, j_idx] = ops.ReduceSum()(x[b, i_idx] * y[b, j_idx])
    # return result
    # print("einsum_func5")
    import torch
    a = torch.einsum('bij,bje->bie', torch.from_numpy(x.asnumpy()), torch.from_numpy(y.asnumpy()))
    return mindspore.Tensor(a.numpy())

def expand_as(tensor, target):
    tensor_shape = np.shape(tensor)
    target_shape = np.shape(target)
    assert tensor_shape[1] == target_shape[1]
    t1 = time.time()
    expanded_tensor = np.zeros_like(target)
    t2 = time.time()
    # print("expand_as 1:{}".format(t2-t1))
    # 将原始张量的每个元素复制到扩展后的张量中
    expanded_tensor[:] = tensor.numpy().tolist()[0]
    # for i in range(target_shape[0]):
    #     expanded_tensor[i] = tensor.numpy().tolist()[0]
    t3 = time.time()
    # print("expand_as 2:{}".format(t3-t2))
    return mindspore.tensor(expanded_tensor.tolist(), dtype=mindspore.float32)

def expand(tensor, dim):
    shape = list(tensor.shape)
    shape[0] = dim
    expanded_tensor = np.zeros(shape)
    for i in range(dim):
        expanded_tensor[i] = tensor.tolist()[0]
    return mindspore.tensor(expanded_tensor.tolist())

def batched_index_select(values, indices):
    indices = ops.Cast()(indices, mindspore.int32)
    last_dim = values.shape[-1]
    # return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))
    # a = mindspore.numpy.tile(ops.ExpandDims()(indices[:, :, None], -1),(1, 1, last_dim))
    size = mindspore.Tensor(np.array([-1, -1, last_dim]), mindspore.int32)
    a = ops.expand(indices[:, :, None], size)
    return ops.GatherD()(values, 1, a)



def sort_key_val(t1, t2, dim=-1):
    # values, indices = t1.sort(dim=dim)
    sort = ops.Sort(axis=dim)
    values, indices = sort(t1)
    t2 = t2.expand_as(t1)
    # t2 = expand_as(t2, t1)
    # return values, t2.gather(dim, indices)
    return values, ops.GatherD()(t2, dim, indices)

def apply_rotary_pos_emb(qk, sinu_pos):
    '''
     to be implemented
    '''
    return

class LSHAttention(Cell):
    def __init__( self,
                  dropout = 0.,
                  bucket_size = 64,
                  n_hashes = 8,
                  causal = False,
                  allow_duplicate_attention = True,
                  attend_across_buckets = True,
                  rehash_each_round = True,
                  drop_for_hash_rate = 0.,
                  random_rotations_per_head = False,
                  return_attn = False):
        super().__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = Dropout(p=dropout)
        self.dropout_for_hash = Dropout(p=drop_for_hash_rate)

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.bucket_size = bucket_size

        self.n_hashes = n_hashes

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

        # will expend extra computation to return attention matrix
        self._return_attn = return_attn

        # cache buckets for reversible network, reported by authors to make Reformer work at depth
        self._cache = {}

    @cache_method_decorator('_cache', 'buckets', reexecute=True)
    def hash_vectors(self, n_buckets, vecs):
        batch_size = vecs.shape[0]
        # device = vecs.device

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert n_buckets % 2 == 0
        rot_size = n_buckets

        rotations_shape = (     # (1, 64, 4, 24)
            batch_size if self._random_rotations_per_head else 1,
            vecs.shape[-1],
            self.n_hashes if self._rehash_each_round else 1,
            rot_size // 2)

        # random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1, -1)
        random_rotations = expand(np.random.normal(0, 1, rotations_shape), batch_size)
        dropped_vecs = self.dropout_for_hash(vecs)

        # rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)
        rotated_vecs = einsum_func1(dropped_vecs, random_rotations) # (256, 64, 4, 24)

        # print("Has vector1:{}".format(t12-t11))
        if self._rehash_each_round:
            # rotated_vectors size [batch,n_hash,seq_len,buckets]
            rotated_vecs = cat( (rotated_vecs, -rotated_vecs), dim=-1)
            # buckets = torch.argmax(rotated_vecs, dim=-1)
            buckets = rotated_vecs.argmax(axis=-1)
        else:   # 应该没执行到这一步
            print("检查")
            exit()
            rotated_vecs = cat( (rotated_vecs, -rotated_vecs), dim=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            # rotated_vecs = torch.squeeze(rotated_vecs, 1)
            rotated_vecs = ops.squeeze(rotated_vecs, axis=1)
            # bucket_range = torch.arange(rotated_vecs.shape[-1], device=device)
            bucket_range = ops.arange(rotated_vecs.shape[-1]).astype(mindspore.float32)
            bucket_range = ops.reshape(bucket_range, (1, -1))
            bucket_range = bucket_range.expand_as(rotated_vecs)
            # bucket_range = expand_as(bucket_range, rotated_vecs)
            _, buckets = sort_key_val(rotated_vecs, bucket_range, dim=-1)
            # buckets size [batch size, seq_len, buckets]
            # buckets = buckets[... , -self.n_hashes:].transpose(1, 2)
            buckets = buckets[... , -self.n_hashes:].transpose(0, 2, 1)

        # buckets is now (self.n_hashes, seq_len). Next we add offsets so that
        # bucket numbers from different hashing rounds don't overlap.

        # t12 = time.time()
        offsets = ops.arange(self.n_hashes).astype(mindspore.float32)
        # t13 = time.time()
        # print("Has vector2:{}".format(t13 - t12))

        offsets = ops.reshape(offsets * n_buckets, (1, -1, 1))
        buckets = ops.reshape(buckets + offsets, (batch_size, -1,))
        return buckets

    def construct(self, qk, v, query_len = None, input_mask = None, input_attn_mask = None, pos_emb = None, **kwargs):
        # batch_size, seqlen, dim, device = *qk.shape, qk.device

        batch_size, seqlen, dim = qk.shape
        query_len = default(query_len, seqlen)
        is_reverse = kwargs.pop('_reverse', False)
        depth = kwargs.pop('_depth', None)
        assert seqlen % (self.bucket_size * 2) == 0, f'Sequence length ({seqlen}) needs to be divisible by target bucket size  x 2 - {self.bucket_size * 2}'
        n_buckets = seqlen // self.bucket_size


        buckets = self.hash_vectors(n_buckets, qk, key_namespace=depth, fetch=is_reverse, set_cache=self.training)

        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seqlen
        total_hashes = self.n_hashes


        # ticker = torch.arange(total_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
        a = ops.arange(total_hashes * seqlen)

        ticker = ops.ExpandDims()(a, 0)

        t4 = time.time()
        # ticker = expand_as(ticker, buckets)
        ticker = ticker.expand_as(buckets)


        buckets_and_t = seqlen * buckets + (ticker % seqlen)
        # buckets_and_t = buckets_and_t.detach()
        buckets_and_t = Parameter(buckets_and_t, requires_grad=False)
        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        # _, undo_sort = sticker.sort(dim=-1)
        _, undo_sort = sticker.sort(axis = -1)
        del ticker


        # sbuckets_and_t = sbuckets_and_t.detach()
        sbuckets_and_t = Parameter(sbuckets_and_t, requires_grad=False)
        # sticker = sticker.detach()
        # undo_sort = undo_sort.detach()
        sticker = Parameter(sticker, requires_grad=False)
        undo_sort = Parameter(undo_sort, requires_grad=False)
        if exists(pos_emb):
            # 跳过了
            qk = apply_rotary_pos_emb(qk, pos_emb)

        st = (sticker % seqlen)
        sqk = batched_index_select(qk, st)
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        chunk_size = total_hashes * n_buckets
        bq_t = bkv_t = ops.reshape(st, (batch_size, chunk_size, -1))
        bqk = ops.reshape(sqk, (batch_size, chunk_size, -1, dim))
        bv = ops.reshape(sv, (batch_size, chunk_size, -1, dim))

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk    # (256, 192, 4, 64)
        # bk = F.normalize(bqk, p=2, dim=-1).type_as(bq)
        l2_normalize = ops.L2Normalize(axis=-1)
        bk = l2_normalize(bqk)

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = cat( (x[:, -1:, ...], x[:, :-1, ...]), dim=1)
            return cat( (x, x_extra), dim=2)

        bk = look_one_back(bk)  # (256, 192, 8, 64)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)

        # Dot-product attention.
        # dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5)    # dim int:64
        dots = einsum_func2(bq, bk) * (dim ** -0.5)

        masked_value = max_neg_value(dots)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            # input_attn_mask = F.pad(input_attn_mask, (0, seqlen - input_attn_mask.shape[-1], 0, seqlen - input_attn_mask.shape[-2]), value=True)
            input_attn_mask = ops.pad(input_attn_mask, padding= (0, seqlen - input_attn_mask.shape[-1], 0, seqlen - input_attn_mask.shape[-2]), value=True)
            dot_attn_indices = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            input_attn_mask = input_attn_mask.reshape(batch_size, -1)
            dot_attn_indices = dot_attn_indices.reshape(batch_size, -1)
            # mask = input_attn_mask.gather(1, dot_attn_indices).reshape_as(dots)
            # mask = ops.gather(input_params = input_attn_mask, input_indices = dot_attn_indices, axis = 1).reshape_as(dots)
            mask = ops.GatherD()(input_attn_mask, 1, dot_attn_indices).reshape_as(dots)
            # dots.masked_fill_(~mask, masked_value)
            dots = ops.masked_fill(dots, ~mask, masked_value)
            del mask

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            # input_mask = F.pad(input_mask, (0, seqlen - input_mask.shape[1]), value=True)
            input_mask = ops.pad(input_mask, padding=(0, seqlen - input_mask.shape[1]), value=True)
            # mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
            # mq = ops.gather(input_params = input_mask, input_indices=dot_attn_indices, axis=1).reshape((batch_size, chunk_size, -1))
            mq = ops.GatherD()(input_mask, 1, dot_attn_indices).reshape((batch_size, chunk_size, -1))
            mkv = look_one_back(mq)
            mask = mq[:, :, :, None] * mkv[:, :, None, :]
            # dots.masked_fill_(~mask, masked_value)
            dots = ops.masked_fill(dots, ~mask, masked_value)
            del mask

        # Causal masking
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :]
            if seqlen > query_len:
                mask = mask & (bkv_t[:, :, None, :] < query_len)
            # dots.masked_fill_(mask, masked_value)
            dots = ops.masked_fill(dots, mask, masked_value)
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :] # (256, 192, 4, 8)都是TrueFalse
        # dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        dots = ops.masked_fill(dots, self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets: # 跳过了
            bq_buckets = bkv_buckets = ops.reshape(sbuckets_and_t // seqlen, (batch_size, chunk_size, -1))
            bkv_buckets = look_one_back(bkv_buckets)
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            # dots.masked_fill_(bucket_mask, masked_value)
            dots = ops.masked_fill(dots, bucket_mask, masked_value)
            del bucket_mask

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention: # 跳过了
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % chunk_size
            if not self._attend_across_buckets:
                locs1 = buckets * chunk_size + locs1
                locs2 = buckets * chunk_size + locs2
            locs = cat(
                (ops.reshape(locs1, (batch_size, total_hashes, seqlen)),
                ops.reshape(locs2, (batch_size, total_hashes, seqlen))), 1).transpose((0, 2, 1))

            slocs = batched_index_select(locs, st)
            b_locs = ops.reshape(slocs, (batch_size, chunk_size, -1, 2 * total_hashes))

            b_locs1 = b_locs[:, :, :, None, :total_hashes]

            # bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, total_hashes))
            bq_locs = ops.expand(b_locs1, mindspore.Tensor(np.array(b_locs.shape[:3] + (2, total_hashes)), mindspore.int32))
            bq_locs = ops.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)

            dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :])
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(total_hashes * batch_size))
            # dup_counts = dup_counts.detach()
            dup_counts = Parameter(dup_counts, requires_grad=False)
            assert dup_counts.shape == dots.shape
            # dots = dots - torch.log(dup_counts + 1e-9)
            dots = dots - ops.log(dup_counts + 1e-9)
            del dup_counts


        # Softmax.
        dots_logsumexp = ops.logsumexp(dots, axis=-1, keep_dims=True)
        # dots = torch.exp(dots - dots_logsumexp).type_as(dots)
        dots = ops.exp(dots - dots_logsumexp)
        dropped_dots = self.dropout(dots)

        # bo = torch.einsum('buij,buje->buie', dropped_dots, bv)
        bo = einsum_func3(dropped_dots, bv)
        so = ops.reshape(bo, (batch_size, -1, dim))
        slogits = ops.reshape(dots_logsumexp, (batch_size, -1,))


        # unsort logits
        o = batched_index_select(so, undo_sort)
        # logits = slogits.gather(1, undo_sort)
        # logits = ops.gather(input_params=slogits, input_indices=undo_sort, axis=1)
        logits = ops.GatherD()(slogits, 1, undo_sort)

        o = ops.reshape(o, (batch_size, total_hashes, seqlen, dim))
        logits = ops.reshape(logits, (batch_size, total_hashes, seqlen, 1))

        if query_len != seqlen:
            query_slice = (slice(None), slice(None), slice(0, query_len))
            o, logits = o[query_slice], logits[query_slice]

        # probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        probs = ops.exp(logits - ops.logsumexp(logits, axis = 1, keep_dims=True))
        # out = torch.sum(o * probs, dim=1)
        out = o * probs
        out = out.sum(axis = 1)
        # attn = torch.empty(0, device=device)
        attn = mindspore.Tensor([])

        # return unsorted attention weights
        if self._return_attn:
            attn_unsort = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            attn_unsort = attn_unsort.view(batch_size * total_hashes, -1).long()
            unsorted_dots = ops.zeros((batch_size * total_hashes, seqlen * seqlen))
            unsorted_dots.scatter_add_(1, attn_unsort, dots.view_as(attn_unsort))
            del attn_unsort
            unsorted_dots = unsorted_dots.reshape(batch_size, total_hashes, seqlen, seqlen)
            # attn = torch.sum(unsorted_dots[:, :, 0:query_len, :] * probs, dim=1)
            op = ops.ReduceSum(keep_dims=True)
            attn = op(unsorted_dots[:, :, 0:query_len, :] * probs, 1)
        # return output, attention matrix, and bucket distribution

        return out, attn, buckets

class FullQKAttention(Cell):
    def __init__(self, causal = False, dropout = 0.):
        super().__init__()
        self.causal = causal
        self.dropout = Dropout(p=dropout)

    def construct(self, qk, v, query_len = None, input_mask = None, input_attn_mask = None, **kwargs):
        b, seq_len, dim = qk.shape
        query_len = default(query_len, seq_len)
        t = query_len

        q = qk[:, 0:query_len]
        # qk = F.normalize(qk, 2, dim=-1).type_as(q)
        l2_normalize = ops.L2Normalize(axis=-1)
        qk = l2_normalize(qk)

        # dot = torch.einsum('bie,bje->bij', q, qk) * (dim ** -0.5)
        dot = einsum_func4(q, qk) * (dim ** -0.5)

        # qk attention requires tokens not attend to self
        i = ops.arange(t).astype(mindspore.float32)
        dot[:, i, i] = TOKEN_SELF_ATTN_VALUE
        masked_value = max_neg_value(dot)

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            mask = input_mask[:, 0:query_len, None] * input_mask[:, None, :]
            # mask = F.pad(mask, (0, seq_len - mask.shape[-1]), value=True)
            mask = ops.pad(mask, padding=(0, seq_len - mask.shape[-1]), mode='constant', value=True)
            # dot.masked_fill_(~mask, masked_value)
            dot = ops.masked_fill(dot, ~mask, masked_value)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            # input_attn_mask = F.pad(input_attn_mask, (0, seq_len - input_attn_mask.shape[-1]), value=True)
            input_attn_mask = ops.pad(input_attn_mask, padding= (0, seq_len - input_attn_mask.shape[-1]), value=True)
            # dot.masked_fill_(~input_attn_mask, masked_value)
            dot = ops.masked_fill(dot, ~input_attn_mask, masked_value)
        if self.causal:
            # i, j = torch.triu_indices(t, t, 1)
            i, j = mindspore.numpy.triu_indices(n = t, m = t, k = 1)
            dot[:, i, j] = masked_value

        dot = dot.softmax(dim=-1)
        dot = self.dropout(dot)

        # out = torch.einsum('bij,bje->bie', dot, v)
        out = einsum_func5(dot, v)

        # return out, dot, torch.empty(0)
        return out, dot, mindspore.Tensor([])


class LSHSelfAttention(Cell):
    def __init__(self, dim, heads = 8, bucket_size = 64, n_hashes = 8, causal = False, dim_head = None, attn_chunks = 1,
                 random_rotations_per_head = False, attend_across_buckets = True, allow_duplicate_attention = True, num_mem_kv = 0,
                 one_value_head = False, use_full_attn = False, full_attn_thres = None, return_attn = False, post_attn_dropout = 0.,
                 dropout = 0., n_local_attn_heads = 0, **kwargs):
        super().__init__()
        assert dim_head or (dim % heads) == 0, 'dimensions must be divisible by number of heads'
        assert n_local_attn_heads < heads, 'local attention heads must be less than number of heads'

        dim_head = default(dim_head, dim // heads)
        dim_heads = dim_head * heads

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.attn_chunks = default(attn_chunks, 1)

        self.v_head_repeats = (heads if one_value_head else 1)  # 1
        v_dim = dim_heads // self.v_head_repeats

        self.toqk = _Linear(dim, dim_heads, has_bias = False)
        self.tov = _Linear(dim, v_dim, has_bias = False)
        self.to_out = _Linear(dim_heads, dim)

        self.bucket_size = bucket_size
        self.lsh_attn = LSHAttention(bucket_size=bucket_size, n_hashes=n_hashes, causal=causal,
                                     random_rotations_per_head=random_rotations_per_head, attend_across_buckets = attend_across_buckets,
                                     allow_duplicate_attention = allow_duplicate_attention, return_attn = return_attn,
                                     dropout = dropout, **kwargs)
        self.full_attn = FullQKAttention(causal=causal, dropout=dropout)
        self.post_attn_dropout = Dropout(p = post_attn_dropout)

        self.use_full_attn = use_full_attn
        self.full_attn_thres = default(full_attn_thres, bucket_size)

        self.num_mem_kv = num_mem_kv
        # self.mem_kv = nn.Parameter(torch.randn(1, num_mem_kv, dim, requires_grad=True)) if num_mem_kv > 0 else None
        self.mem_kv = Parameter(mindspore.numpy.randn((1, num_mem_kv, dim)), requires_grad=True) if num_mem_kv > 0 else None

        self.n_local_attn_heads = n_local_attn_heads
        # self.local_attn = LocalAttention(window_size=bucket_size * 2, causal=causal, dropout=dropout, shared_qk=True, look_forward=(1 if not causal else 0))
        self.local_attn = None  # not implemented yet
        self.callback = None

    def construct(self, x, keys = None, input_mask = None, input_attn_mask = None, context_mask = None, pos_emb = None, **kwargs):
        # device, dtype = x.device, x.dtype
        dtype = x.dtype
        b, t, e, h, dh, m, l_h = *x.shape, self.heads, self.dim_head, self.num_mem_kv, self.n_local_attn_heads

        # mem_kv = default(self.mem_kv, torch.empty(b, 0, e, dtype=dtype, device=device))
        mem_kv = default(self.mem_kv, mindspore.numpy.empty((b, 0, e)))
        # mem = mem_kv.expand((b, m, -1))
        size = mindspore.Tensor(np.array([b, m, -1]), mindspore.int32)
        mem = ops.expand(mem_kv, size)
        # keys = default(keys, torch.empty(b, 0, e, dtype=dtype, device=device))
        keys = default(keys, mindspore.numpy.empty((b, 0, e)))
        c = keys.shape[1]

        kv_len = t + m + c
        use_full_attn = self.use_full_attn or kv_len <= self.full_attn_thres

        x = cat((x, mem, keys), dim=1)
        qk = self.toqk(x)
        v = self.tov(x) # Tensor(32, 192, 512)
        # v = v.repeat(1, 1, self.v_head_repeats)
        v = mindspore.numpy.tile(v, (1, 1, self.v_head_repeats))

        def merge_heads(v):
            return v.view((b, kv_len, h, -1)).transpose(0, 2, 1, 3)

        def split_heads(v):
            return v.view((b, h, t, -1)).transpose(0, 2, 1, 3)

        merge_batch_and_heads = partial(merge_dims, 0, 1)
        qk, v = map(merge_heads, (qk, v))

        has_local = l_h > 0
        lsh_h = h - l_h

        split_index_fn = partial(split_at_index, 1, l_h)
        (lqk, qk), (lv, v) = map(split_index_fn, (qk, v))
        # print("(lqk, qk, lv, v)", (lqk, qk, lv, v))
        # lqk, qk, lv, v = map(merge_batch_and_heads, (lqk, qk, lv, v))
        qk = merge_dims(0, 1, qk)
        v = merge_dims(0, 1, v)


        masks = {}
        if input_mask is not None or context_mask is not None:
            # default_mask = torch.tensor([True], device=device)
            default_mask = mindspore.tensor([True])
            # i_mask = default(input_mask, default_mask.expand(b, t))
            i_mask = default(input_mask, ops.expand(default_mask, mindspore.Tensor(np.array([b, t]), mindspore.int32)))
            # m_mask = default_mask.expand(b, m)
            size = mindspore.Tensor(np.array([b, m]), mindspore.int32)
            m_mask = ops.expand(default_mask, size)
            # c_mask = default(context_mask, default_mask.expand(b, c))
            c_mask = default(context_mask, ops.expand(default_mask, mindspore.Tensor(np.array([b, c]), mindspore.int32)))
            mask = cat((i_mask, m_mask, c_mask), dim=1)
            mask = merge_batch_and_heads(expand_dim(1, lsh_h, mask))
            masks['input_mask'] = mask

        if input_attn_mask is not None:
            input_attn_mask = merge_batch_and_heads(expand_dim(1, lsh_h, input_attn_mask))
            masks['input_attn_mask'] = input_attn_mask


        attn_fn = self.lsh_attn if not use_full_attn else self.full_attn
        partial_attn_fn = partial(attn_fn, query_len = t, pos_emb = pos_emb, **kwargs)
        attn_fn_in_chunks = process_inputs_chunk(partial_attn_fn, chunks = self.attn_chunks)
        out, attn, buckets = attn_fn_in_chunks(qk, v, **masks)


        if self.callback is not None:
            self.callback(attn.reshape(b, lsh_h, t, -1), buckets.reshape(b, lsh_h, -1))

        if has_local:
            '''
            Not implemented yet
            '''
            print("Not implemented has_local")
            exit()
            # lqk, lv = lqk[:, :t], lv[:, :t]
            # local_out = self.local_attn(lqk, lqk, lv, input_mask=input_mask)
            # local_out = local_out.reshape(b, l_h, t, -1)
            # out = out.reshape(b, lsh_h, t, -1)
            # out = cat((local_out, out), dim=1)

        out = split_heads(out).view(b, t, -1)
        out = self.to_out(out)
        return self.post_attn_dropout(out)