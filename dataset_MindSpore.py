'''
https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset
'''
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from timefeatures import time_features
from mindspore.dataset import GeneratorDataset
import numpy as np
import mindspore
from mindspore import nn
from mindspore.ops import operations as P



class StandardizeColumn():
    def __init__(self, column_name):
        super(StandardizeColumn, self).__init__()
        self.column_name = column_name
        self.mean = P.ReduceMean()
        self.sub = P.Sub()
        self.div = P.Div()
        self.mean_value = None
        self.std_value = None
    def fit(self, x):
        tensor = mindspore.Tensor(x[self.column_name].values).astype(mindspore.float64)
        self.mean_value = self.mean(tensor)
        reduce_mean = self.sub(tensor, self.mean_value)
        self.std_value = self.mean(P.Square()(reduce_mean))
        self.std_value = P.Sqrt()(self.std_value)
    def transform(self, x):
        tensor = mindspore.Tensor(x[self.column_name].values).astype(mindspore.float64)
        normalized_value = self.sub(tensor, self.mean_value)
        normalized_value = self.div(normalized_value, self.std_value)
        return normalized_value


def list2array(batch_list):
    # 将列表中的每个ndarray展平为(96*7,)的形状，并存储到新列表中
    flattened_arrays = [arr.flatten() for arr in batch_list]
    # 将展平后的列表中的所有ndarray合并为一个新的ndarray，其形状为(64, 96*7)
    flattened_array = np.concatenate(flattened_arrays)
    # 使用reshape函数将形状为(64, 96*7)的ndarray转换为(64, 96, 7)的ndarray
    return flattened_array.reshape(len(batch_list), len(batch_list[0]), -1)


class MyIterable:
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, batch_size = None):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.batch_size = batch_size
        # __read_data__()
        # self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if self.data_path[:4] == 'ETTh':
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.data_path[:4] == 'ETTm':
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            print("输入data_path不符合要求")
            exit()

        print("border1s", border1s)
        print("border2s", border2s)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]       # <class 'pandas.core.frame.DataFrame'>, 8640 rows x 7 columns
            # Rewrite
            column_names = train_data.columns
            standardize_modules = []
            for column in column_names:
                scale = StandardizeColumn(column)
                scale.fit(train_data)
                standardize_modules.append(scale.transform(df_data))
            for i, j in enumerate(standardize_modules):
                standardize_modules[i] = j.numpy()
            data = []
            for i in range(len(standardize_modules[0])):
                data.append([j[i] for j in standardize_modules])
            data = np.array(data)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            if self.data_path[:4] == 'ETTm':
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self._index = 0


    def __next__(self):
        if self._index + self.batch_size >= (len(self.data_x) - self.seq_len - self.pred_len + 1):  # drop_last
            raise StopIteration
        else:
            seq_x = []
            seq_y = []
            seq_x_mark = []
            seq_y_mark = []
            for _ in range(self.batch_size):
                s_begin = self._index
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len

                seq_x.append(self.data_x[s_begin:s_end])
                seq_y.append(self.data_y[r_begin:r_end])
                seq_x_mark.append(self.data_stamp[s_begin:s_end])
                seq_y_mark.append(self.data_stamp[r_begin:r_end])
                self._index += 1
            return (list2array(seq_x), list2array(seq_y), list2array(seq_x_mark), list2array(seq_y_mark))

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        # print("self.data_x",len(self.data_x))
        # print("self.seq_len", self.seq_len)
        # print("self.pred_len",  self.pred_len)
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) // self.batch_size





def data_provider(args, flag):
    Data = MyIterable
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq


    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
        batch_size=args.batch_size
    )

    print(flag, len(data_set))

    data_loader = GeneratorDataset(source=data_set, column_names=["col1", "col2", "col3", "col4"], shuffle=False)

    return data_set, data_loader



# import argparse
# parser = argparse.ArgumentParser(description='Transformer')
# parser.add_argument('--embed', type=str, default='timeF',
#                     help='time features encoding, options:[timeF, fixed, learned]')
# parser.add_argument('--task_name', type=str, default='long_term_forecast',
#                     help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
# parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
# parser.add_argument('--freq', type=str, default='h',
#                     help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
# parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
# parser.add_argument('--root_path', type=str, default='./data/',
#                     help='root path of the data file')
# parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
# parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
# parser.add_argument('--label_len', type=int, default=48, help='start token length')
# parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
# parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
# parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
# parser.add_argument('--features', type=str, default='M',
#                     help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
# parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
# args = parser.parse_args()
#
# data_set, data_loader = data_provider(args, 'train')