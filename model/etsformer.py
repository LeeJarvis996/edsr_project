import sys
sys.path.append("..")
import mindspore
import mindspore.ops as ops
from mindspore.nn.cell import Cell
from layer.basic import _Linear, Dropout
from layer.Embed import DataEmbedding
from layer.basic import _Linear
from layer.etsformer_attn import Transform, GrowthLayer, FourierLayer, LevelLayer, Feedforward, DampingLayer


class Encoder(Cell):
    def __init__(self, layers):
        super().__init__()
        # self.layers = nn.ModuleList(layers)
        self.layers = mindspore.nn.CellList(layers)

    def construct(self, res, level, attn_mask=None):
        growths = []
        seasons = []
        for layer in self.layers:
            res, level, growth, season = layer(res, level, attn_mask=None)
            growths.append(growth)
            seasons.append(season)

        return level, growths, seasons


class EncoderLayer(Cell):

    def __init__(self, d_model, nhead, c_out, seq_len, pred_len, k, dim_feedforward=None, dropout=0.1,
                 activation='sigmoid', layer_norm_eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.c_out = c_out
        self.seq_len = seq_len
        self.pred_len = pred_len
        dim_feedforward = dim_feedforward or 4 * d_model
        self.dim_feedforward = dim_feedforward

        self.growth_layer = GrowthLayer(d_model, nhead, dropout=dropout)
        self.seasonal_layer = FourierLayer(d_model, pred_len, k=k)
        self.level_layer = LevelLayer(d_model, c_out, dropout=dropout)

        # Implementation of Feedforward model
        self.ff = Feedforward(d_model, dim_feedforward, dropout=dropout, activation=activation)
        # self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm1 = mindspore.nn.LayerNorm((d_model,), epsilon=layer_norm_eps)
        self.norm2 = mindspore.nn.LayerNorm((d_model,), epsilon=layer_norm_eps)

        self.dropout1 = Dropout(p = dropout)
        self.dropout2 = Dropout(p = dropout)

    def construct(self, res, level, attn_mask=None):
        # print("season_block")
        season = self._season_block(res)
        res = res - season[:, :-self.pred_len]
        # print("growth_block")
        growth = self._growth_block(res)
        res = self.norm1(res - growth[:, 1:])
        res = self.norm2(res + self.ff(res))
        # print("level_layer")
        level = self.level_layer(level, growth[:, :-1], season[:, :-self.pred_len])
        return res, level, growth, season

    def _growth_block(self, x):
        x = self.growth_layer(x)
        return self.dropout1(x)

    def _season_block(self, x):
        x = self.seasonal_layer(x)
        return self.dropout2(x)


class Decoder(Cell):
    def __init__(self, layers):
        super().__init__()
        self.d_model = layers[0].d_model
        self.c_out = layers[0].c_out
        self.pred_len = layers[0].pred_len
        self.nhead = layers[0].nhead

        self.layers = mindspore.nn.CellList(layers)
        self.pred = _Linear(self.d_model, self.c_out)

    def construct(self, growths, seasons):
        growth_repr = []
        season_repr = []

        for idx, layer in enumerate(self.layers):
            growth_horizon, season_horizon = layer(growths[idx], seasons[idx])
            growth_repr.append(growth_horizon)
            season_repr.append(season_horizon)
        growth_repr = sum(growth_repr)
        season_repr = sum(season_repr)
        return self.pred(growth_repr.astype(mindspore.float32)), self.pred(season_repr.astype(mindspore.float32))


class DecoderLayer(Cell):
    def __init__(self, d_model, nhead, c_out, pred_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.c_out = c_out
        self.pred_len = pred_len
        self.growth_damping = DampingLayer(pred_len, nhead, dropout=dropout)
        self.dropout1 = Dropout(p = dropout)

    def construct(self, growth, season):
        growth_horizon = self.growth_damping(growth[:, -1:])
        growth_horizon = self.dropout1(growth_horizon)

        seasonal_horizon = season[:, -self.pred_len:]
        return growth_horizon, seasonal_horizon


class Etsformer(Cell):
    """
    Paper link: https://arxiv.org/abs/2202.01381
    """

    def __init__(self, args):
        super(Etsformer, self).__init__()
        self.task_name = args.task_name
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = args.seq_len
        else:
            self.pred_len = args.pred_len

        assert args.e_layers == args.d_layers, "Encoder and decoder layers must be equal"

        # Embedding
        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq,
                                           args.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    args.d_model, args.n_heads, args.enc_in, args.seq_len, self.pred_len, args.top_k,
                    dim_feedforward=args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation,
                ) for _ in range(args.e_layers)
            ]
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    args.d_model, args.n_heads, args.c_out, self.pred_len,
                    dropout=args.dropout,
                ) for _ in range(args.d_layers)
            ],
        )
        self.transform = Transform(sigma=0.2)

        if self.task_name == 'classification':
            # self.act = torch.nn.functional.gelu
            self.act= ops.gelu
            self.dropout = Dropout(p = args.dropout)
            self.projection = _Linear(args.d_model * args.seq_len, args.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.training:
            x_enc = self.transform.transform(x_enc)
        # print("enc_embedding")
        res = self.enc_embedding(x_enc, x_mark_enc)
        # print("encoder")
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)
        # print("decoder")
        growth, season = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds

    def construct(self, src, src_mark, tgt, tgt_mark, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(src, src_mark, tgt, tgt_mark)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None