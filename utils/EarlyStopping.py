import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Modify from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py"""
    def __init__(self, patience=10, delta=0, lr_patience=5, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.counter = 0
        self.lr_patience = lr_patience
        self.best_score = None
        self.early_stop = False
        self.reduce_lr = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.reduce_lr = False
            if self.counter == self.lr_patience:
                self.reduce_lr = True
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter    = 0
            self.reduce_lr  = False
            self.best_score = score
