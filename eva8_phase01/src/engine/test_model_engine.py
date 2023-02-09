#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Engine to train Test Models'''


#imports
import torch
import torch.nn as nn
import torchmetrics
#   script imports
from dataloaders.test_model_dataset import MNISTDataLoader
from engine.base.base_engine import BaseEngine
from model.test_model_3a import TestModel3A
from model.test_model_4a import TestModel4A
from model.test_model_4b import TestModel4B
from model.test_model_4c import TestModel4C
from model.test_model_4d import TestModel4D
from model.test_model_5 import TestModel5

#imports

# classes
class TestModelEngine(BaseEngine):
  '''Engine to train LeNet models'''

  def __init__(self, hparams):
    super().__init__(hparams)

  def _init_train_dataloader(self):
    self.train_ds = MNISTDataLoader(
      root=str(self.hparams.data_path),
      train=True,
      augments=self.hparams.do_augment
    )


  def _init_test_dataloader(self):
    self.test_ds = MNISTDataLoader(
      root=str(self.hparams.data_path),
      train=False,
      augments=self.hparams.do_augment
    )


  def _init_model(self):
    if self.hparams.model_name == 'test_model_3a':
      self.model = TestModel3A(num_classes=self.hparams.num_classes)
    elif self.hparams.model_name == 'test_model_4a':
      self.model = TestModel4A(num_classes=self.hparams.num_classes)
    elif self.hparams.model_name == 'test_model_4b':
      self.model = TestModel4B(num_classes=self.hparams.num_classes)
    elif self.hparams.model_name == 'test_model_4c':
      self.model = TestModel4C(num_classes=self.hparams.num_classes)
    elif self.hparams.model_name == 'test_model_4d':
      self.model = TestModel4D(num_classes=self.hparams.num_classes)
    elif self.hparams.model_name in ['test_model_5_bn', 'test_model_5_gn', 'test_model_5_ln']:
      self.model = TestModel5(num_classes=self.hparams.num_classes, norm_type=self.hparams.norm_type)


  def _init_loss_function(self):
    if self.hparams.loss_function == 'CrossEntropyLoss':
      self.loss_function = {
        'loss_mnist': nn.CrossEntropyLoss()
      }
    elif self.hparams.loss_function == 'NLLLoss':
      self.loss_function = {
        'loss_mnist': nn.NLLLoss(reduction='sum')
      }


  def _init_optimizer(self):
    if self.hparams.optimizer == 'SGD':
      self.optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=self.hparams.learning_rate,
        momentum=0.9
      )
    elif self.hparams.optimizer == 'Adam':
      self.optimizer = torch.optim.Adam(
        self.model.parameters(),
        lr=self.hparams.learning_rate
      )


  def _init_scheduler(self):
    if self.hparams.lr_scheduler == 'None':
      pass
    if self.hparams.lr_scheduler == 'StepLR':
      self.scheduler = torch.optim.lr_scheduler.StepLR(
        self.optimizer,
        **self.hparams.lr_scheduler_args
      )
    elif self.hparams.lr_scheduler == 'MultiStepLR':
      self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        self.optimizer,
        **self.hparams.lr_scheduler_args
      )

  def _init_metrics(self):
    self.metrics = {}
    for each in self.hparams.metrics:
      if each == 'accuracy':
        self.metrics['m_mnist_accuracy'] = torchmetrics.Accuracy(
          task='multiclass',
          num_classes=self.hparams.num_classes
        )
        self.metrics['test_m_mnist_accuracy'] = torchmetrics.Accuracy(
          task='multiclass',
          num_classes=self.hparams.num_classes
        )
      elif each == 'f1score':
        self.metrics['m_mnist_f1score'] = torchmetrics.F1Score(
          task='multiclass',
          num_classes=self.hparams.num_classes
        )
        self.metrics['test_m_mnist_f1score'] = torchmetrics.F1Score(
          task='multiclass',
          num_classes=self.hparams.num_classes
        )

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
