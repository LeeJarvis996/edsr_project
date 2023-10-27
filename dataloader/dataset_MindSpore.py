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
from dataloader.m4 import M4Dataset, M4Meta
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
        elif self.data_path[:13] == 'exchange_rate':
            # 10 years for training, 4 years for validation, 5 years for testing
            border1s = [0, 12 * 30 * 10 - self.seq_len, 12 * 30 * 10 + 12 * 30 * 4 - self.seq_len]
            border2s = [12 * 30 * 10, 12 * 30 * 10 + 12 * 30 * 4, 12 * 30 * 10 + + 12 * 30 * 4 + 12 * 30 * 5]
        elif self.data_path[:11] == 'electricity':
            # 2 years for training, 1 year for validation, 1 year for testing
            border1s = [0, 12 * 30 * 24 * 2 - self.seq_len, 12 * 30 * 24 * 2 + 12 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24 * 2, 12 * 30 * 24 * 2 + 12 * 30 * 24, 12 * 30 * 24 * 2 + 12 * 30 * 24 * 2]
        elif self.data_path[:16] == 'national_illness':
            # 10 years for training, 3 years for validation, 4 years for testing
            border1s = [0, 12 * 4 * 10 - self.seq_len, 12 * 4 * 10 + 12 * 4 * 3 - self.seq_len]
            border2s = [12 * 4 * 10, 12 * 4 * 10 + 12 * 4 * 3, 12 * 4 * 10 + + 12 * 4 * 3 + 12 * 4 * 4]
        elif self.data_path[:7] == 'traffic':
            # 1 year for training, 4 months for validation, 5 months for testing
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 9 * 30 * 24]
        elif self.data_path[:7] == 'weather':
            # 200 days for training, 60 days for validation, 80 days for testing
            border1s = [0, 200 * 24 * 6 - self.seq_len, 200 * 24 * 6 + 60 * 24 * 6 - self.seq_len]
            border2s = [200 * 24 * 6, 200 * 24 * 6 + 60 * 24 * 6, 200 * 24 * 6 + 140 * 24 * 6]
        else:
            print("输入data_path不符合要求")
            exit()

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
        # print("border1:{}, border2:{}".format(border1, border2))
        self.data_x = data[border1:border2]
        # print("len(self.data_x", len(self.data_x))
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



class Dataset_M4:
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path=None,
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly', batch_size = None):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.batch_size = batch_size
        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag
        self._index = 0
        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __next__(self):
        if self._index + self.batch_size >=  len(self.timeseries):  # drop_last
        # if self._index + self.batch_size >= 132:  # drop_last
            raise StopIteration
        else:
            insample_lst = []
            outsample_lst = []
            insample_mask_lst = []
            outsample_mask_lst = []
            for _ in range(self.batch_size):
                insample = np.zeros((self.seq_len, 1))
                insample_mask = np.zeros((self.seq_len, 1))
                outsample = np.zeros((self.pred_len + self.label_len, 1))
                outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset
                sampled_timeseries = self.timeseries[self._index]
                cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                              high=len(sampled_timeseries),
                                              size=1)[0]

                insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
                insample[-len(insample_window):, 0] = insample_window
                insample_mask[-len(insample_window):, 0] = 1.0
                outsample_window = sampled_timeseries[
                                   cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
                outsample[:len(outsample_window), 0] = outsample_window
                outsample_mask[:len(outsample_window), 0] = 1.0

                insample_lst.append(insample)
                outsample_lst.append(outsample)
                insample_mask_lst.append(insample_mask)
                outsample_mask_lst.append(outsample_mask)
                self._index += 1

            return (insample_lst, outsample_lst, insample_mask_lst, outsample_mask_lst)


    def __len__(self):
        return len(self.timeseries) // self.batch_size - 1

    def __iter__(self):
        self._index = 0
        return self

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


def data_provider(args, flag):
    if args.task_name == 'short_term_forecast':
        Data = Dataset_M4
    elif args.task_name == 'long_term_forecast':
        Data = MyIterable
    else:
        print("Invalid task name")
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
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

