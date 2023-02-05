#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Console Logger implementation'''


#imports
#   script imports
from callbacks.callback_handler import Callback
#imports


# classes
class ConsoleLogger(Callback):
  '''Console Logger implementation'''

  def on_train_begin(self, **kwargs):
    print('Started Training')

  def on_train_end(self, **kwargs):
    print('Finished Training')

  def on_epoch_end(self, **kwargs):
    print_st = f"Train Epoch: {kwargs['epoch'] + 1}"
    print_st += f" | Loss: {kwargs['loss_mnist']:.4f}"
    print_st += f" | Accuracy: {kwargs['m_mnist_accuracy'] * 100:.2f}%"

    print(print_st)

  def on_test_epoch_end(self, **kwargs):
    print_st = f"Test  Epoch: {kwargs['epoch'] + 1}"
    print_st += f" | Loss: {kwargs['test_loss_mnist']:.4f}"
    print_st += f" | Accuracy: {kwargs['test_m_mnist_accuracy'] * 100:.2f}%"

    print(print_st)

    print('--------------------------------')


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
