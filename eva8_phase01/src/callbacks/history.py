#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Create a history object to record model training data'''

import os

#imports
from matplotlib import pyplot as plt
from callbacks.callback_handler import Callback
#   script imports
#imports


# functions
def create_training_charts(epochs, metrics:dict, title:str, save_folder:str):
  fig, ax = plt.subplots(figsize=(20, 15))
  for key, value in metrics.items():
    ax.plot(epochs, value, label=key)
  ax.set_xlabel('Epochs', fontsize=15)
  ax.set_ylabel(title, fontsize=15)
  ax.set_title(title, fontsize=20)
  ax.legend()

  fig.savefig(
    fname=os.path.join(save_folder, f'{title}.png'),
    format='png'
  )

# functions

# classes
class History(Callback):
  '''Create a history object to record model training data'''

  def __init__(self):
    super().__init__()
    self.history = {}
    self.add_keys('epoch')

  def add_keys(self, key):
    self.history[key] = []

  def on_train_end(self, **kwargs):
    # loss = dict(
    #   (key, value) for key, value in self.history.items() if 'loss' in key and 'test_' not in key
    # )
    # metrics = dict(
    #     (key, value) for key, value in self.history.items() if 'm_' in key and 'test_' not in key
    # )
    # create_training_charts(
    #   self.history['epoch'],
    #   metrics=loss,
    #   title='Loss',
    #   save_folder=kwargs['save_folder']
    # )
    # create_training_charts(
    #   self.history['epoch'],
    #   metrics=metrics,
    #   title='Metrics',
    #   save_folder=kwargs['save_folder']
    # )
    pass

  def on_epoch_end(self, **logs):
    logs = logs if logs is not None else {}

    # append logs of loss & metrics to history
    for k, v in logs.items():
      self.history[k].append(v)

  def on_test_epoch_end(self, **logs):
    logs = logs if logs is not None else {}

    # append logs of loss & metrics to history
    for k, v in logs.items():
      if k != 'epoch':
        self.history[k].append(v)


# classes


# main
def main():
  pass


# if main script
if __name__ == '__main__':
  main()
