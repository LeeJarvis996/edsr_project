import os
from model import autoformer, transformer, informer, reformer, etsformer, pyraformer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Autoformer': autoformer.Autoformer,
            'Transformer': transformer.Transformer,
            'Informer': informer.Informer,
            'Reformer': reformer.Reformer,
            'Etsformer': etsformer.Etsformer,
            'Pyraformer': pyraformer.Pyraformer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        print('We only support cpu.')
        return

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass