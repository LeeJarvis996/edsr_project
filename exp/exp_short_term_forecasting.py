import sys
sys.path.append("..")
from dataloader.dataset_MindSpore import data_provider
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary
from dataloader.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
import os
import time
import warnings
import numpy as np
import pandas as pd
import mindspore
from mindspore import value_and_grad
from mindspore import Parameter
from mindspore import nn
from mindspore import Parameter
import mindspore.ops as ops
import mindspore.common.tensor as Tensor
from utils.tools import Monitor, visual
import time

warnings.filterwarnings('ignore')

class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)

    def _build_model(self):
        self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
        self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
        self.args.label_len = self.args.pred_len
        self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        model = self.model_dict[self.args.model](self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = nn.Adam(params=self.model.trainable_params(), learning_rate=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss(reduction='mean')
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    def forward_fn(self, batch_x, dec_inp, batch_x_mark, batch_y_mark, batch_y):
        output = self.model(src = batch_x, src_mark = None, tgt = dec_inp, tgt_mark = None)
        f_dim = -1 if self.args.features == 'MS' else 0
        output = output[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
        batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:]
        loss = self._select_criterion(self.args.loss)(batch_x, self.args.frequency_map, output, batch_y, batch_y_mark)
        return loss, output

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        control = Monitor(patience=self.args.patience)
        epoch_loss = []
        time_all = []
        time_now = time.time()
        for epoch in range(self.args.train_epochs):
            if control.early_stop(): break
            train_loss = []
            epoch_time = time.time()
            for iter, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if iter >= 20:
                    break
                self.model.set_train()
                dec_inp = ops.zeros_like(batch_y[:, -self.args.pred_len:, :], dtype=mindspore.float64)
                dec_inp = mindspore.ops.cat([batch_y[:, :self.args.label_len, :], dec_inp], axis=1)

                self.model.set_train()
                grad_fn = value_and_grad(self.forward_fn, grad_position=None,
                                         weights=self._select_optimizer().parameters, has_aux=True)
                (loss, _), grads = grad_fn(batch_x, dec_inp, batch_x_mark, batch_y_mark, batch_y)
                self._select_optimizer()(grads)
                train_loss.append(loss.numpy().item())
                print("Epoch:{}| iter:{}| loss:{}".format(epoch + 1, iter, loss.item()))

                if (iter + 1) % 20 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(iter + 1, epoch + 1, np.average(train_loss)))
                    # speed = (time.time() - time_now) / iter_count
                    # iter_count = 0
                    time_all.append(time.time() - time_now)
                    epoch_loss.append(np.average(train_loss))
                    time_now = time.time()

            train_loss = np.average(train_loss)
            print("Epoch: {}, Steps: {} | Train Loss: {} | Time Cost: {}".format(
                epoch + 1, len(train_loader), train_loss, time.time() - epoch_time))
            # df = pd.DataFrame({'loss': epoch_loss, 'time': time_all})
            # df.to_csv('record_mindspore.csv')

            mindspore.save_checkpoint(self.model, "./ckpts/model{}.ckpt".format(epoch))

            print("*" * 10, "Validation", "*" * 10)
            control.add(self.vali(train_loader, vali_loader, self._select_criterion(self.args.loss)))
            time_now = time.time()

        print("Loading the Best Model: epoch = {}".format(control.best_epoch()))
        param_dict = mindspore.load_checkpoint("./ckpts/model{}.ckpt".format(control.best_epoch()))
        param_not_load, _ = mindspore.load_param_into_net(self.model, param_dict)

        return self.model

    def vali(self, train_loader, vali_loader, criterion):
        self.model.set_train(False)
        x, _ = train_loader.source.last_insample_window()
        y = vali_loader.source.timeseries
        x = Tensor(x, dtype=mindspore.float32)
        x = ops.ExpandDims()(x, -1)
        B, _, C = x.shape
        dec_inp = ops.zeros((B, self.args.pred_len, C), dtype=mindspore.float32)
        dec_inp = ops.concat([x[:, -self.args.label_len:, :], dec_inp], 1)
        # encoder - decoder
        outputs = ops.zeros((B, self.args.pred_len, C), dtype=mindspore.float32)
        id_list = np.arange(0, B, 500)  # validation set size
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[Tensor(id_list[i]):Tensor(id_list[i + 1]), :, :] = Parameter(self.model(x[Tensor(id_list[i]):Tensor(id_list[i + 1])], None,
                                                                  dec_inp[Tensor(id_list[i]):Tensor(id_list[i + 1])],
                                                                  None), requires_grad=False)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        pred = outputs
        true = Tensor(np.array(y))
        batch_y_mark = ops.ones(true.shape, dtype=mindspore.float32)
        x = Parameter(x, requires_grad=False)
        loss = criterion(x[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        return loss



    def test(self, setting, test=0):
        self.model.set_train(False)
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        x, _ = train_loader.source.last_insample_window()
        y = test_loader.source.timeseries
        x = Tensor(x, dtype=mindspore.float32)
        x = ops.ExpandDims()(x, -1)

        if test:
            print('loading model')
            param_dict = mindspore.load_checkpoint("./ckpts/model{}.ckpt".format(10000))
            param_not_load, _ = mindspore.load_param_into_net(self.model, param_dict)

        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        B, _, C = x.shape
        dec_inp = ops.zeros((B, self.args.pred_len, C), dtype=mindspore.float32)
        dec_inp = ops.concat([x[:, -self.args.label_len:, :], dec_inp], 1)

        # encoder - decoder
        outputs = ops.zeros((B, self.args.pred_len, C), dtype=mindspore.float32)
        id_list = np.arange(0, B, 1)
        id_list = np.append(id_list, B)

        for i in range(len(id_list) - 1):
            outputs[Tensor(id_list[i]):Tensor(id_list[i + 1]), :, :] = Parameter(self.model(x[Tensor(id_list[i]):Tensor(id_list[i + 1])], None,
                                                                  dec_inp[Tensor(id_list[i]):Tensor(id_list[i + 1])],
                                                                  None), requires_grad=False)

            if id_list[i] % 1000 == 0:
                print(id_list[i])

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        outputs = Parameter(outputs, requires_grad=False).asnumpy()

        preds = outputs
        x = Parameter(x, requires_grad=False).asnumpy()

        print('test shape:', preds.shape)

        # result save
        folder_path = './results/m4_results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = pd.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.source.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

        print(self.args.model)
        file_path = './results/m4_results/' + self.args.model + '/'
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            m4_summary = M4Summary(file_path, self.args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            print('smape:', smape_results)
            print('mape:', mape)
            print('mase:', mase)
            print('owa:', owa_results)
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')
        return

