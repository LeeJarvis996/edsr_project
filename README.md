This is just a temporary version. This Time-series Prediction Benchmark is under the leadership of Professor Lei CHEN.
Great thanks to [TSlib](https://github.com/thuml/Time-Series-Library/tree/main).

🎉 **NEWS**: 
- 2023-10-27 Support the Short-term-forcasting task.
- 2023-10-16 Add several benchmark datasets.
- 2023-10-09 Support [ETSformer](https://github.com/thuml/Time-Series-Library/blob/main/models/ETSformer.py).
- 2023-09-23 Support [Autoformer](https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py).
- 2023-09-23 Support [Pyraformer](https://github.com/thuml/Time-Series-Library/blob/main/models/Pyraformer.py).
- 2023-09-22 Support [Informer](https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py).
- 2023-09-15 Support [Reformer](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py).
- 2023-09-02 Support [Transformer](https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py).
- 2023-08-28 Support the Long-term-forcasting task.
- 2023-08-27 Finished Dataloader




# Contents
- [Installation](#Installation)
- [Data](#Data)
- [How to run](#Run)


# The directory structure of the code
```shell
.
├── ckpts            # Models will be saved here during training.
├── data             # Benchmark dataset.
├── dataloader       # Dataloader.
├── exp              # Long-term and short-term forecasting tasks
├── layer            # These codes are modified from mindspore.nn.layer, including the internal implementations of the model.
├── model     		 # Framework of the models.
├── results          # Results.
├── utils            # Other functions.
├── exp_.py          # Main function 
└── README.md
```

# Installation
For now, we only support CPU version MindSpore. We are executing the code using Python 3.8 on a Windows x64 platform. You can refer to [here](https://www.mindspore.cn/install). run:
```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.1.1/MindSpore/cpu/x86_64/mindspore-2.1.1-cp38-cp38-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# Data
Please download the data from the [Google Drive](https://drive.google.com/file/d/1-nC6xR3L4G7JgARqJjlcNjE82in5jAjB/view?usp=sharing).

# Run
```shell
python exp.py  \
    --model 'Transformer' \
    --train_epoch 10
```



Other parameters
- ``model``: Could choose from ['Transformer', 'Informer', 'Reformer', 'Pyraformer', 'Autoformer'].
- ``patience``: For early stop.
- ``batch_size``: Batch_size.
- ``learning_rate``: Learning rate of the optimizer.
- ``features``: Forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate.
- ``enc_in``: Encoder input size.
- ``dec_in``: Decoder input size.
- ``c_out``: Output size.
- ``dropout``: Dropout rate.
- ``embed``: time features encoding, options:[timeF, fixed, learned].

# Dataset configurations
For long-term forecasting tasks:


<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="3">ETTh</th>
      <th colspan="3">Exchange_rate</th>
      <th colspan="3">Electricity</th>
      <th colspan="3">National_illness</th>
      <th colspan="3">Traffic</th>
      <th colspan="3">Weather</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>args.features</td>
      <td>'M'</td>
      <td>'S'</td>
      <td>'MS'</td>
      <td>'M'</td>
      <td>'S'</td>
      <td>'MS'</td>
      <td>'M'</td>
      <td>'S'</td>
      <td>'MS'</td>
      <td>'M'</td>
      <td>'S'</td>
      <td>'MS'</td>
      <td>'M'</td>
      <td>'S'</td>
      <td>'MS'</td>
      <td>'M'</td>
      <td>'S'</td>
      <td>'MS'</td>
    </tr>
    <tr>
      <td>args.target</td>
      <td>'OT'</td>
      <td>'OT'</td>
      <td>'OT'</td>
      <td>'OT'</td>
      <td>'OT'</td>
      <td>'OT'</td>
      <td>'OT'</td>
      <td>'OT'</td>
      <td>'OT'</td>
      <td>'OT'</td>
      <td>'OT'</td>
      <td>'OT'</td>
      <td>'OT'</td>
      <td>'OT'</td>
      <td>'OT'</td>
      <td>'OT'</td>
      <td>'OT'</td>
      <td>'OT'</td>
    </tr>
    <tr>
      <td>args.enc_in</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
      <td>8</td>
      <td>1</td>
      <td>8</td>
      <td>321</td>
      <td>1</td>
      <td>321</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
      <td>862</td>
      <td>1</td>
      <td>862</td>
      <td>21</td>
      <td>1</td>
      <td>21</td>
    </tr>
    <tr>
      <td>args.dec_in</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
      <td>8</td>
      <td>1</td>
      <td>8</td>
      <td>321</td>
      <td>1</td>
      <td>321</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
      <td>862</td>
      <td>1</td>
      <td>862</td>
      <td>21</td>
      <td>1</td>
      <td>21</td>
    </tr>
    <tr>
      <td>args.c_out</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>321</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>862</td>
      <td>1</td>
      <td>1</td>
      <td>21</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

For short-term forecasting tasks, we support four models: Transformer, Autoformer, Pyraformer, and Informer. Please configure args.features as 'S'.
# Outstanding issues
- For Reformer, there is no CPU-based Mindspore equivalent of the PyTorch torch.einsum() function. Consequently, we continue to utilize the PyTorch version of this function in our code for its superior performance.(layers/reformer_attn.py) If you prefer not to use PyTorch, we also offer our own custom time-inefficient function, which can be found in the commented-out code at the same location.
- For Autoformer, ops.roll does not support CPU, and therefore we use the numpy instead.(layers/autoformer_attn.py)
- For ETSformer,
    - since the gradient is not supported for complex type multiplication currently, we have to do multiplication with nd.array format.(layers/autoformer_attn.py, layers/etsformer_attn.py)
    - since the mindspore.ops.FFTWithSize is not same as the torch.rfft/irfft, we use numpy instead.(layers/etsformer_attn.py)
- For now, we only provide long-term-forcast-task. We will support short-term-forcast-term in the future.
