#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Callbacnk handler for pytorch'''


#imports
from collections.abc import Iterable
from typing import Any
#   script imports
#imports

def listify(p: Any) -> Iterable:
  if p is None:
    p = []
  elif not isinstance(p, Iterable):
    p = [p]
  return p


# classes
class Callback:
  '''Abstract class for callbacks'''
  def __init__(self): pass
  def on_train_begin(self, **kwargs): pass
  def on_train_end(self, **kwargs): pass
  def on_epoch_begin(self, **kwargs): pass
  def on_epoch_end(self, **kwargs): pass
  def on_batch_begin(self, **kwargs): pass
  def on_batch_end(self, **kwargs): pass
  def on_test_begin(self, **kwargs): pass
  def on_test_end(self, **kwargs): pass
  def on_test_epoch_begin(self, **kwargs): pass
  def on_test_epoch_end(self, **kwargs): pass
  def on_test_batch_begin(self, **kwargs): pass
  def on_test_batch_end(self, **kwargs): pass
  def run_predictions(self, **kwargs): pass


class CallbackHandler:
  '''Callbacnk handler for pytorch'''

  def __init__(self, callbacks):
    super(CallbackHandler, self).__init__()
    self.callbacks = listify(callbacks)

  def on_train_begin(self, **kwargs):
    for callback in self.callbacks:
      callback.on_train_begin(**kwargs)

  def on_train_end(self, **kwargs):
    for callback in self.callbacks:
      callback.on_train_end(**kwargs)

  def on_epoch_begin(self, **kwargs):
    for callback in self.callbacks:
      callback.on_epoch_begin(**kwargs)

  def on_epoch_end(self, **kwargs):
    for callback in self.callbacks:
      callback.on_epoch_end(**kwargs)

  def on_batch_begin(self, **kwargs):
    for callback in self.callbacks:
      callback.on_batch_begin(**kwargs)

  def on_batch_end(self, **kwargs):
    for callback in self.callbacks:
      callback.on_batch_end(**kwargs)

  def on_test_begin(self, **kwargs):
    for callback in self.callbacks:
      callback.on_test_begin(**kwargs)

  def on_test_end(self, **kwargs):
    for callback in self.callbacks:
      callback.on_test_end(**kwargs)

  def on_test_epoch_begin(self, **kwargs):
    for callback in self.callbacks:
      callback.on_test_epoch_begin(**kwargs)

  def on_test_epoch_end(self, **kwargs):
    for callback in self.callbacks:
      callback.on_test_epoch_end(**kwargs)

  def on_test_batch_begin(self, **kwargs):
    for callback in self.callbacks:
      callback.on_test_batch_begin(**kwargs)

  def on_test_batch_end(self, **kwargs):
    for callback in self.callbacks:
      callback.on_test_batch_end(**kwargs)

  def run_predictions(self, **kwargs):
    for callback in self.callbacks:
      callback.run_predictions(**kwargs)

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
