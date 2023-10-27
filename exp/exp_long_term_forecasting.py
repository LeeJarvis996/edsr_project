import sys
sys.path.append("..")
from dataloader.dataset_MindSpore import data_provider
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
import mindspore.ops as ops
from utils.tools import Monitor
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
    def _build_model(self):
        model = self.model_dict[self.args.model](self.args)

        # 查看所有参数
        total_params = 0
        for param in model.trainable_params():
            print(param)
            total_params += np.prod(param.shape)
        print(f"总参数数量: {total_params}")
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = nn.Adam(params=self.model.trainable_params(), learning_rate=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss(reduction='mean')
        return criterion

    def forward_fn(self, batch_x, dec_inp, batch_x_mark, batch_y_mark, batch_y):
        output = self.model(src = batch_x, src_mark = batch_x_mark, tgt = dec_inp, tgt_mark = batch_y_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
        output = output[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
        loss = self._select_criterion()(output, batch_y)
        return loss, output

    def vali(self, vali_loader):
        vali_loss = []
        for iter, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            dec_inp = ops.zeros_like(batch_y[:, -self.args.pred_len:, :], dtype=mindspore.float64)
            dec_inp = ops.cat([batch_y[:, :self.args.label_len, :], dec_inp], axis=1)

            self.model.set_train(False)
            output = self.model(src = batch_x, src_mark = batch_x_mark, tgt = dec_inp, tgt_mark = batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            output = output[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
            loss = self._select_criterion()(output, batch_y)
            vali_loss.append(loss.numpy().item())

        vali_loss = np.average(vali_loss)
        print("Validation| Steps: {}| Validation Loss: {}".format(len(vali_loader), vali_loss))
        return vali_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        import time
        control = Monitor(patience=self.args.patience)
        epoch_loss = []
        time_all = []
        time_now = time.time()
        for epoch in range(self.args.train_epochs):
            if control.early_stop(): break
            train_loss = []
            epoch_time = time.time()
            for iter, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                dec_inp = ops.zeros_like(batch_y[:, -self.args.pred_len:, :], dtype=mindspore.float64)
                dec_inp = mindspore.ops.cat([batch_y[:, :self.args.label_len, :], dec_inp], axis=1)

                self.model.set_train()
                # optimizer = nn.SGD(model.trainable_params(), learning_rate=args.learning_rate)
                grad_fn = value_and_grad(self.forward_fn, grad_position=None, weights=self._select_optimizer().parameters, has_aux=True)
                (loss, _), grads = grad_fn(batch_x, dec_inp, batch_x_mark, batch_y_mark, batch_y)
                self._select_optimizer()(grads)
                train_loss.append(loss.numpy().item())
                print("Epoch:{}| iter:{}| loss:{}".format(epoch + 1, iter, loss.item()))

                if (iter + 1) % 20 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(iter + 1, epoch + 1, np.average(train_loss)))
                    # speed = (time.time() - time_now) / iter_count
                    # iter_count = 0
                    time_all.append(time.time() - time_now)
                    epoch_loss.append(np.average(train_loss))
                    time_now = time.time()

            train_loss = np.average(train_loss)
            print("Epoch: {}, Steps: {} | Train Loss: {} | Time Cost: {}".format(
                epoch + 1, len(train_loader), train_loss, time.time() - epoch_time))
            df = pd.DataFrame({'loss': epoch_loss, 'time': time_all})
            df.to_csv('record_mindspore.csv')

            mindspore.save_checkpoint(self.model, "./ckpts/model{}.ckpt".format(epoch))

            print("*" * 10, "Validation", "*" * 10)
            control.add(self.vali(vali_loader))
            time_now = time.time()

        print("Loading the Best Model: epoch = {}".format(control.best_epoch()))
        param_dict = mindspore.load_checkpoint("./ckpts/model{}.ckpt".format(control.best_epoch()))
        param_not_load, _ = mindspore.load_param_into_net(self.model, param_dict)

        return self.model

    def test(self, setting, test=0):
        print("*" * 10, "TEST", "*" * 10)
        test_data, test_loader = self._get_data(flag='test')
        test_loss = []
        preds = []
        trues = []
        self.model.set_train(False)
        for iter, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            dec_inp = ops.zeros_like(batch_y[:, -self.args.pred_len:, :], dtype=mindspore.float64)
            dec_inp = mindspore.ops.cat([batch_y[:, :self.args.label_len, :], dec_inp], axis=1)
            output = self.model(src=batch_x, src_mark=batch_x_mark, tgt=dec_inp, tgt_mark=batch_y_mark)
            f_dim = -1 if self.args.features == 'MS' else 0
            output = output[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            outputs = Parameter(output, requires_grad=False).asnumpy()
            batch_y = Parameter(batch_y, requires_grad=False).asnumpy()
            pred = outputs
            true = batch_y

            preds.append(pred)
            trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return