#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Base Engine to train models
'''


#imports
import torch
#   script imports
from callbacks.history import History
from torch.utils.data import DataLoader
from torchsummary import summary

#imports


# classes
class BaseEngine:
  '''
  Base Engine to train models
  '''

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

  def _init_model(self):
    raise NotImplementedError

  def _init_optimizer(self):
    raise NotImplementedError

  def _init_loss_function(self):
    raise NotImplementedError

  def _init_metrics(self):
    raise NotImplementedError

  def _init_scheduler(self):
    self.scheduler = None

  def _show_summary(self):
    print('--------------------------------')
    print('Model Summary')
    print('--------------------------------')
    print(summary(self.model, input_size=(1, 28, 28)))
    print('--------------------------------')

  def _init_callbacks(self):
    self.history = History()

    # print(list(self.loss_function.keys()))

    # create history recorder callback
    for each in list(self.loss_function.keys()) + list(self.metrics.keys()):
      self.history.add_keys(each)

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

  def train_step(self):
    raise NotImplementedError

  def test_step(self):
    raise NotImplementedError

  def _init_logger(self):
    raise NotImplementedError

  def _seed_it_all(self):
    pass

  def train(self):
    print(f'Training {self.hparams.model_name}...')
    self.history.on_train_start()
    for epoch in range(self.hparams.epochs):
      # get epoch loss
      epoch_loss = dict([(key, []) for key in self.loss_function.keys()])
      epoch_accuracy = dict([(key, 0) for key in self.metrics.keys()])
      # create and train batch
      for batch in self.train_loader:
        loss_values = self.train_step(batch, train=True)

        # create logs for update history
        for key, value in loss_values.items():
          epoch_loss[key].append(value)

      for key, value in self.metrics.items():
        epoch_accuracy[key] = value.compute().item()

      for key, value in epoch_loss.items():
        epoch_loss[key] = sum(epoch_loss[key])/len(epoch_loss[key])

      self.history.on_epoch_end(epoch, {**epoch_loss, **epoch_accuracy})
      print(self.create_print_statement_per_epoch(epoch))


    self.history.on_train_end(str(self.hparams.charts))
    print('Training Ended...')

  def test(self):
    print(f'Testing {self.hparams.model_name}...')
    for batch in self.test_loader:
      self.train_step(batch)

    print(self.create_print_statement_per_epoch(epoch=0))
    print('Testing Ended...')


  def create_print_statement_per_epoch(self, epoch):
    print_st = f'Epoch: {str(epoch+1).zfill(len(str(self.hparams.epochs)))}'
    for metric, metric_obj in self.metrics.items():
      print_st += f' | {metric}: {metric_obj.compute()*100:.2f}%'
      metric_obj.reset()

    return print_st

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
