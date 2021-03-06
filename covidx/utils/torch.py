import torch
import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given number of consecutive epochs"""
    def __init__(self, model, patience=1, delta=1e-4):
        """
        Instantiate an EarlyStopping object.

        :param model: The model.
        :param patience: The number of consecutive epochs to wait.
        :param delta: The minimum change of the monitored quantity.
        """
        self.model = model
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.should_stop = False
        self.counter = 0
        self.best_state = None

    @property
    def best_found(self):
        return self.counter == 0 and self.best_loss != np.inf

    def __call__(self, loss):
        """
        Call the object.

        :param loss: The validation loss measured.
        """
        # Check if an improved of the loss happened
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0
            self.best_state = self.model.state_dict()
        else:
            self.counter += 1

        # Check if the training should stop
        if self.counter >= self.patience:
            self.should_stop = True

    def load_state_dict(self, state_dict):
        self.patience = state_dict['patience']
        self.delta = state_dict['delta']
        self.best_loss = state_dict['best_loss']
        self.should_stop = state_dict['should_stop']
        self.counter = state_dict['counter']
        self.best_state = state_dict['best_state']

    def state_dict(self):
        return {
            'patience': self.patience,
            'delta': self.delta,
            'best_loss': self.best_loss,
            'should_stop': self.should_stop,
            'counter': self.counter,
            'best_state': self.best_state
        }


class RunningAverageMetric:
    """Running (batched) average metric"""

    def __init__(self, batch_size):
        """
        Initialize a running average metric object.

        :param batch_size: The batch size.
        """
        self.batch_size = batch_size
        self.metric_accumulator = 0.0
        self.n_metrics = 0

    def __call__(self, x):
        """
        Accumulate a metric.

        :param x: The metric value.
        """
        self.metric_accumulator += x
        self.n_metrics += 1

    def average(self):
        """
        Get the metric average.

        :return: The metric average.
        """
        return self.metric_accumulator / (self.n_metrics * self.batch_size)


def get_optimizer(optimizer):
    return {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'rmsprop': torch.optim.RMSprop
    }[optimizer]
