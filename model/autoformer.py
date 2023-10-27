import sys
sys.path.append("..")
import mindspore
import mindspore.ops as ops
from mindspore.nn.cell import Cell
from layer.basic import _Linear, Dropout
from layer.Embed import DataEmbedding_wo_pos
from layer.autoformer_attn import series_decomp, AutoCorrelationLayer, AutoCorrelation


class EncoderLayer(Cell):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.conv1 = mindspore.nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, has_bias = False)
        self.conv2 = mindspore.nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, has_bias = False)

        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        # self.dropout = nn.Dropout(dropout)
        self.dropout = Dropout(p=dropout)
        self.activation = ops.relu if activation == "relu" else ops.gelu

    def construct(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(0, 2, 1))))
        y = self.dropout(self.conv2(y).transpose(0, 2, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(Cell):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        # self.attn_layers = nn.ModuleList(attn_layers)
        # self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.attn_layers = mindspore.nn.CellList(attn_layers)
        self.conv_layers = mindspore.nn.CellList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def construct(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class Decoder(Cell):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        # self.layers = nn.ModuleList(layers)
        self.layers = mindspore.nn.CellList(layers)
        self.norm = norm_layer
        self.projection = projection

    def construct(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend


class DecoderLayer(Cell):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.conv1 = mindspore.nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, has_bias = False)
        self.conv2 = mindspore.nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, has_bias = False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        # self.dropout = nn.Dropout(dropout)
        self.dropout = Dropout(p=dropout)

        # self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
        #                             padding_mode='circular', bias=False)
        self.projection = mindspore.nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    pad_mode='pad', has_bias=False)
        # self.activation = F.relu if activation == "relu" else F.gelu
        self.activation = ops.relu if activation == "relu" else ops.gelu


    def construct(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(0, 2, 1))))
        y = self.dropout(self.conv2(y).transpose(0, 2, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.transpose(0, 2, 1)).transpose(0, 2, 1)
        return x, residual_trend


class Autoformer(Cell):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """
    def __init__(self, args):
        super(Autoformer, self).__init__()
        self.task_name = args.task_name
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention

        # Decomp
        kernel_size = args.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(args.enc_in, args.d_model, args.embed, args.freq,
                                                  args.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, args.factor, attention_dropout=args.dropout,
                                        output_attention=args.output_attention),
                        args.d_model, args.n_heads),
                    args.d_model,
                    args.d_ff,
                    moving_avg=args.moving_avg,
                    dropout=args.dropout,
                    activation=args.activation
                ) for l in range(args.e_layers)
            ],
            norm_layer = my_Layernorm(args.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding_wo_pos(args.dec_in, args.d_model, args.embed, args.freq,
                                                      args.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, args.factor, attention_dropout=args.dropout,
                                            output_attention=False),
                            args.d_model, args.n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, args.factor, attention_dropout=args.dropout,
                                            output_attention=False),
                            args.d_model, args.n_heads),
                        args.d_model,
                        args.c_out,
                        args.d_ff,
                        moving_avg=args.moving_avg,
                        dropout=args.dropout,
                        activation=args.activation,
                    )
                    for l in range(args.d_layers)
                ],
                norm_layer = my_Layernorm(args.d_model),
                projection = _Linear(args.d_model, args.c_out, has_bias=True)
            )


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # mean = torch.mean(x_enc, dim=1).unsqueeze(
        #     1).repeat(1, self.pred_len, 1)
        mean = mindspore.numpy.tile(ops.ExpandDims()(ops.mean(x_enc, axis=1), 1), (1, self.pred_len, 1))
        # zeros = torch.zeros([x_dec.shape[0], self.pred_len,
        #                      x_dec.shape[2]], device=x_enc.device)
        # zeros = ops.Zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]])
        zeros = ops.zeros((x_dec.shape[0], self.pred_len, x_dec.shape[2]))
        seasonal_init, trend_init = self.decomp(x_enc)

        # decoder input
        # trend_init = torch.cat(
        #     [trend_init[:, -self.label_len:, :], mean], dim=1)
        # seasonal_init = torch.cat(
        #     [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        trend_init = ops.concat([trend_init[:, -self.label_len:, :], mean.astype('Float32')], 1)
        seasonal_init = ops.concat([seasonal_init[:, -self.label_len:, :].astype('Float32'), zeros], 1)

        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out


    def construct(self, src, src_mark, tgt, tgt_mark, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(src, src_mark, tgt, tgt_mark)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None


class my_Layernorm(Cell):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        # self.layernorm = nn.LayerNorm(channels)
        self.layernorm = mindspore.nn.LayerNorm((channels,))
    def construct(self, x):
        x_hat = self.layernorm(x)
        # bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        bias = mindspore.numpy.tile(ops.ExpandDims()(ops.mean(x_hat, axis=1), 1), (1, x.shape[1], 1))
        return x_hat - bias

