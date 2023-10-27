import matplotlib.pyplot as plt

class Monitor():
    def __init__(self, patience):
        super(Monitor, self).__init__()
        self.patience = patience
        self.count = 0
        self.vali_loss = []
        self.min_loss = 100000

    def early_stop(self):
        if self.count >= self.patience:
            return True
        else:
            return False

    def add(self, loss):
        self.vali_loss.append(loss)
        if loss < self.min_loss:
            self.min_loss = loss
            self.count = 0
        else:
            self.count += 1
            print("Count:{} |Patience:{}".format(self.count, self.patience))

    def best_epoch(self):
        min_value = min(self.vali_loss)
        min_index = self.vali_loss.index(min_value)
        return min_index

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')