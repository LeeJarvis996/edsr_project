import argparse
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Informer')
    parser.add_argument('--model', type=str, default='Etsformer',
                        help='model name, options: [Reformer, Transformer, Informer, Pyraformer, Autoformer, Etsformer]')
    parser.add_argument('--patience', type=int, default=50, help='early stop')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--task_name', type=str, default='short_term_forecast',
                        help='task name, options:[short_term_forecast, short_term_forecast]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    # parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    # parser.add_argument('--root_path', type=str, default='./data/long_term_forecast/ETTh/', help='root path of the data file')
    # parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--root_path', type=str, default='./data/short_term_forecast/m4/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='Hourly-train.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    '''
    If features == M, (enc_in, dec_in, c_out) = (7,7,7);
    elif features == S, (enc_in, dec_in, c_out) = (1,1,1)
    elif features == MS, (enc_in, dec_in, c_out) = (7,7,1)
    '''
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--loss', type=str, default='SMAPE', help='loss function')
    args = parser.parse_args()


    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}'.format(
        args.task_name,
        args.model,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil)
    exp = Exp(args)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)


    # loss_fn = nn.MSELoss(reduction='mean')
    # def forward_fn(batch_x, dec_inp, batch_x_mark, batch_y_mark, batch_y):
    #     output = model(src = batch_x, src_mark = batch_x_mark, tgt = dec_inp, tgt_mark = batch_y_mark)
    #     f_dim = -1 if args.features == 'MS' else 0
    #     output = output[:, -args.pred_len:, f_dim:]
    #     batch_y = batch_y[:, -args.pred_len:, f_dim:]
    #     loss = loss_fn(output, batch_y)
    #     return loss, output
    #
    # class Monitor():
    #     def __init__(self, patience):
    #         super(Monitor, self).__init__()
    #         self.patience = patience
    #         self.count = 0
    #         self.vali_loss = []
    #         self.min_loss = 100000
    #     def early_stop(self):
    #         if self.count >= self.patience: return True
    #         else: return False
    #     def add(self, loss):
    #         self.vali_loss.append(loss)
    #         if loss < self.min_loss:
    #             self.min_loss = loss
    #             self.count = 0
    #         else:
    #             self.count += 1
    #             print("Count:{} |Patience:{}".format(self.count, self.patience))
    #     def best_epoch(self):
    #         min_value = min(self.vali_loss)
    #         min_index = self.vali_loss.index(min_value)
    #         return min_index
    #
    #
    # def display_params(net):
    #     total_params = 0
    #     for param in net.trainable_params():
    #         print(param)
    #         total_params += np.prod(param.shape)
    #     print(f"总参数数量: {total_params}")
    #
    #
    # train_set, train_loader = data_provider(args, 'train')
    # vali_set, vali_loader = data_provider(args, 'val')
    # test_set, test_loader = data_provider(args, 'test')
    # if args.model == 'Transformer':
    #     model = Transformer(args = args)
    # elif args.model == 'Reformer':
    #     model = Reformer(args = args)
    # elif args.model == 'Informer':
    #     model = Informer(args = args)
    # elif args.model == 'Pyraformer':
    #     model = Pyraformer(args = args)
    # elif args.model == 'Autoformer':
    #     model = Autoformer(args = args)
    # elif args.model == 'Etsformer':
    #     model = Etsformer(args=args)
    #
    # display_params(model)
    #
    # def vali(vali_loader, model):
    #     vali_loss = []
    #     for iter, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
    #         dec_inp = ops.zeros_like(batch_y[:, -args.pred_len:, :], dtype=mindspore.float64)
    #         dec_inp = mindspore.ops.cat([batch_y[:, :args.label_len, :], dec_inp], axis=1)
    #
    #         model.set_train(False)
    #         output = model(src = batch_x, src_mark = batch_x_mark, tgt = dec_inp, tgt_mark = batch_y_mark)
    #
    #         f_dim = -1 if args.features == 'MS' else 0
    #         output = output[:, -args.pred_len:, f_dim:]
    #         batch_y = batch_y[:, -args.pred_len:, f_dim:]
    #         loss = loss_fn(output, batch_y)
    #         vali_loss.append(loss.numpy().item())
    #
    #     vali_loss = np.average(vali_loss)
    #     print("Validation| Steps: {}| Validation Loss: {}".format(len(vali_loader), vali_loss))
    #     return vali_loss
    #
    # import time
    # control = Monitor(patience=args.patience)
    # epoch_loss = []
    # time_all = []
    # time_now = time.time()
    # for epoch in range(args.train_epochs):
    #     if control.early_stop():break
    #     train_loss = []
    #     epoch_time = time.time()
    #     for iter, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
    #         dec_inp = ops.zeros_like(batch_y[:, -args.pred_len:, :], dtype=mindspore.float64)
    #         dec_inp = mindspore.ops.cat([batch_y[:, :args.label_len, :], dec_inp], axis=1)
    #
    #         model.set_train()
    #         # optimizer = nn.SGD(model.trainable_params(), learning_rate=args.learning_rate)
    #         optimizer = nn.Adam(params=model.trainable_params(), learning_rate=args.learning_rate)
    #         grad_fn = value_and_grad(forward_fn, grad_position=None, weights=optimizer.parameters, has_aux=True)
    #         (loss, _), grads = grad_fn(batch_x, dec_inp, batch_x_mark, batch_y_mark, batch_y)
    #         optimizer(grads)
    #         train_loss.append(loss.numpy().item())
    #         print("Epoch:{}| iter:{}| loss:{}".format(epoch+1, iter, loss.item()))
    #
    #         if (iter + 1) % 20 == 0:
    #             print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(iter + 1, epoch + 1, np.average(train_loss)))
    #             # speed = (time.time() - time_now) / iter_count
    #             # iter_count = 0
    #             time_all.append(time.time() - time_now)
    #             epoch_loss.append(np.average(train_loss))
    #             time_now = time.time()
    #
    #     train_loss = np.average(train_loss)
    #     print("Epoch: {}, Steps: {} | Train Loss: {} | Time Cost: {}".format(
    #         epoch + 1, len(train_loader), train_loss, time.time()-epoch_time))
    #     df = pd.DataFrame({'loss': epoch_loss, 'time': time_all})
    #     df.to_csv('record_mindspore.csv')
    #
    #     mindspore.save_checkpoint(model, "./ckpts/model{}.ckpt".format(epoch))
    #
    #     print("*" * 10, "Validation", "*" * 10)
    #     control.add(vali(vali_loader, model))
    #     time_now = time.time()
    #
    #
    #
    # print("Loading the Best Model: epoch = {}".format(control.best_epoch()))
    # param_dict = mindspore.load_checkpoint("./ckpts/model{}.ckpt".format(control.best_epoch()))
    # param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
    #
    # # test
    # print("*"*10, "TEST", "*"*10)
    # test_loss = []
    # for iter, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #     dec_inp = ops.zeros_like(batch_y[:, -args.pred_len:, :], dtype=mindspore.float64)
    #     dec_inp = mindspore.ops.cat([batch_y[:, :args.label_len, :], dec_inp], axis=1)
    #
    #     model.set_train(False)
    #     output = model(src = batch_x, src_mark = batch_x_mark, tgt = dec_inp, tgt_mark = batch_y_mark)
    #     f_dim = -1 if args.features == 'MS' else 0
    #     output = output[:, -args.pred_len:, f_dim:]
    #     batch_y = batch_y[:, -args.pred_len:, f_dim:]
    #     loss = loss_fn(output, batch_y)
    #     test_loss.append(loss.numpy().item())
    # test_loss = np.average(test_loss)
    # print("Test Loss: {}".format(test_loss))