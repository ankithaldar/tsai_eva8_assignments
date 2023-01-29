#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Base Engine to train models'''


#imports
import torch
#   script imports
from callbacks.history import History
from callbacks.neptune_callback import NeptuneCallback
from callbacks.console_logger import ConsoleLogger
from callbacks.callback_handler import CallbackHandler
from torch.utils.data import DataLoader
from torchsummary import summary

#imports


# classes
class BaseEngine:
  '''Base Engine to train models'''
  def __init__(self, hparams):
    self.hparams = hparams

    self._init_train_dataloader()
    self._init_test_dataloader()

    self._init_model()
    self._init_loss_function()
    self._init_metrics()
    self._init_callbacks()

    self.setup()

  def _init_train_dataloader(self):
    self.train_ds = None

  def _init_test_dataloader(self):
    self.test_ds = None

  def _init_valid_dataloader(self):
    self.valid_ds = None

  def _init_model(self):
    raise NotImplementedError

  def _init_optimizer(self):
    raise NotImplementedError

  def _init_loss_function(self):
    raise NotImplementedError

  def _init_metrics(self):
    raise NotImplementedError

  def _init_scheduler(self):
    raise NotImplementedError

  def _init_callbacks(self):
    self.callback = []

    history = History()
    # create history recorder callback
    for each in list(self.loss_function.keys()) + list(self.metrics.keys()):
      history.add_keys(each)
      history.add_keys(f'test_{each}')

    self.callback.append(history)

    for each in self.hparams.callback_list:
      if each == 'logging':
        self._init_logger()

    self.callbacks = CallbackHandler(self.callback)

  def _init_logger(self):
    if self.hparams.logger_name is not None:
      logger = NeptuneCallback(
        project_name=self.hparams.logger_init_params['project_name'],
        project_params=self.hparams.model_params
      )
      self.callback.append(logger)

      logger = ConsoleLogger()
      self.callback.append(logger)

  def _seed_it_all(self):
    pass

  def _show_summary(self):
    print('--------------------------------')
    print('Model Summary')
    print(summary(self.model, input_size=(1, 28, 28)))

  def setup(self):
    self._init_distribution()
    self._show_summary()

  def _init_distribution(self):
    # Check and Assign GPU if available else assign CPU
    self.device = torch.device(
      'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    # enables benchmark mode in cudnn.
    torch.backends.cudnn.benchmark = True

    # create train dataloeader
    if self.train_ds:
      self.train_loader = DataLoader(
        self.train_ds,
        batch_size=self.hparams.batch_size,
        pin_memory=True,
        # pin_menory_device=self.device,
        **self.hparams.train_args
      )

    # create test dataloader
    if self.test_ds:
      self.test_loader = DataLoader(
        self.test_ds,
        batch_size=self.hparams.batch_size,
        pin_memory=True,
        # pin_menory_device=self.device,
        **self.hparams.test_args
      )

    # Move model to device
    self.model = self.model.to(self.device)

    print('Training on GPU' if self.model.cuda() else 'Training on CPU')

    # initialze optimizer
    self._init_optimizer()
    if self.hparams.lr_scheduler:
      self._init_scheduler()

    for metrics in self.metrics.keys():
      self.metrics[metrics] = self.metrics[metrics].to(self.device)

  def train_step(self, batch, train:bool=True):
    image, label = [i.to(self.device) for i in batch]

    pred_label = self.model(image)
    loss_mnist = self.loss_function['loss_mnist'](pred_label, label)

    if train:
      self.optimizer.zero_grad()
      loss_mnist.backward()
      self.optimizer.step()

    for metric, metric_obj in self.metrics.items():
      if 'mnist' in metric:
        b_acc = metric_obj(pred_label.argmax(dim=1), label)

    if train:
      self.callbacks.on_batch_end(
        loss_mnist=loss_mnist.item(),
        m_mnist_accuracy=b_acc
      )
      return {
        'loss_mnist': loss_mnist.item()
      }
    else:
      self.callbacks.on_test_batch_end(
        loss_mnist=loss_mnist.item(),
        m_mnist_accuracy=b_acc
      )
      return {
        'test_loss_mnist': loss_mnist.item()
      }

  def training(self, epoch):
    # self.callbacks.on_train_begin()

    # get epoch loss
    epoch_loss = dict([(key, []) for key in self.loss_function.keys()])
    epoch_metrics = dict([(key, 0) for key in self.metrics.keys()])

    self.model.train()
    self.callbacks.on_epoch_begin()

    for b_idx, batch in enumerate(self.train_loader):
      self.callbacks.on_batch_begin()

      loss_values = self.train_step(batch, train=True)

      # create logs for update history
      for key, value in loss_values.items():
        epoch_loss[key].append(value)

    for key, value in self.metrics.items():
      epoch_metrics[key] = value.compute().item()

    for key, value in epoch_loss.items():
      epoch_loss[key] = sum(epoch_loss[key])/len(epoch_loss[key])

    self.callbacks.on_epoch_end(
      epoch=epoch,
      **epoch_loss,
      **epoch_metrics
    )

  def testing(self, epoch):
    epoch_loss = dict([(f'test_{key}', []) for key in self.loss_function.keys()])
    epoch_metrics = dict([(f'test_{key}', 0) for key in self.metrics.keys()])

    self.model.eval()
    self.callbacks.on_test_epoch_begin()

    with torch.no_grad():
      for b_idx, batch in enumerate(self.test_loader):
        self.callbacks.on_test_batch_begin()

        loss_values = self.train_step(batch, train=False)

        # create logs for update history
        for key, value in loss_values.items():
          epoch_loss[key].append(value)

    for key, value in self.metrics.items():
      epoch_metrics[key] = value.compute().item()

    for key, value in epoch_loss.items():
      epoch_loss[key] = sum(epoch_loss[key])/len(epoch_loss[key])

    self.callbacks.on_test_epoch_end(
      epoch=epoch,
      **epoch_loss,
      **epoch_metrics
    )

  def fit(self):
    for epoch in range(self.hparams.epochs):
      self.training(epoch=epoch)
      self.testing(epoch=epoch)

    self.callbacks.on_train_end(save_folder=self.hparams.charts)

# classes


# functions
def function_name():
  pass
# functions


# main
def main():
  pass


# if main script
if __name__ == '__main__':
  main()
