This is just a temporary version. This Time-series Prediction Benchmark is under the leadership of Professor Lei CHEN.
Great thanks to [TimeNet repo](https://github.com/thuml/Time-Series-Library/tree/main).

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
- [How to run](#Run)

# The directory structure of the code
```shell
.
├── ckpts            # Models will be saved here during training.
├── data             # Benchmark dataset.
├── layer            # This code closely resembles mindspore.nn.layer, with only a very limited number of functions being modified.             		
├── dataset_MindSpore.py            # Dataloader.
├── timefeatures.py                 # Preprocess data, which will be called by the dataloader.
├── exp.py                          # Main function 
└── README.md
```

# Installation
For now, we only support CPU version MindSpore. We are executing the code using Python 3.8 on a Windows x64 platform. You can refer to [here](https://www.mindspore.cn/install). run:
```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.1.1/MindSpore/cpu/x86_64/mindspore-2.1.1-cp38-cp38-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

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

# Outstanding issues
- For Reformer, there is no CPU-based Mindspore equivalent of the PyTorch torch.einsum() function. Consequently, we continue to utilize the PyTorch version of this function in our code for its superior performance.(layers/reformer_attn.py) If you prefer not to use PyTorch, we also offer our own custom time-inefficient function, which can be found in the commented-out code at the same location.
- For Autoformer, ops.roll does not support CPU, and therefore we use the numpy instead.(layers/autoformer_attn.py)
- For ETSformer,
    - since the gradient is not supported for complex type multiplication currently, we have to do multiplication with nd.array format.(layers/autoformer_attn.py, layers/etsformer_attn.py)
    - since the mindspore.ops.FFTWithSize is not same as the torch.rfft/irfft, we use numpy instead.(layers/etsformer_attn.py)
- For now, we only provide long-term-forcast-task. We will support short-term-forcast-term in the future.
