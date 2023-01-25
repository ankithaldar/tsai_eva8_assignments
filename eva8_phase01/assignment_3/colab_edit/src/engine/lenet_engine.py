#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Engine to train LeNet models'''


#imports
import torch
import torch.nn as nn
import torchmetrics
#   script imports
from dataloaders.mnist_dataset import MNISTDataLoader
from engine.base.base_engine import BaseEngine
from model.lenet import LeNet

#imports


# classes
class LeNetEngine(BaseEngine):
  '''Engine to train LeNet models'''

  def __init__(self, hparams):
    super().__init__(hparams)

  def _init_train_dataloader(self):
    self.train_ds = MNISTDataLoader(
      root=str(self.hparams.data_path),
      train=True
    )


  def _init_test_dataloader(self):
    self.test_ds = MNISTDataLoader(
      root=str(self.hparams.data_path),
      train=False
    )


  def _init_model(self):
    if self.hparams.model_name == 'LeNet':
      self.model = LeNet(num_classes=self.hparams.num_classes)


  def _init_loss_function(self):
    if self.hparams.loss_function == 'CrossEntropyLoss':
      self.loss_function = {
        'loss_mnist': nn.CrossEntropyLoss(),
        'loss_sum': nn.CrossEntropyLoss()
      }


  def _init_optimizer(self):
    if self.hparams.optimizer == 'SGD':
      self.optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=self.hparams.learning_rate,
        momentum=0
      )
    elif self.hparams.optimizer == 'Adam':
      self.optimizer = torch.optim.Adam(
        self.model.parameters(),
        lr=self.hparams.learning_rate
      )


  def _init_scheduler(self):
    if self.hparams.lr_scheduler == 'StepLR':
      self.scheduler = torch.optim.lr_scheduler.StepLR(
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
        self.metrics['m_sum_accuracy'] = torchmetrics.Accuracy(
          task='multiclass',
          num_classes=2*self.hparams.num_classes - 1
        )
      elif each == 'f1score':
        self.metrics['m_mnist_f1score'] = torchmetrics.F1Score(
          task='multiclass',
          num_classes=self.hparams.num_classes
        )
        self.metrics['m_sum_f1score'] = torchmetrics.F1Score(
          task='multiclass',
          num_classes=2*self.hparams.num_classes - 1
        )


  def train_step(self, batch, train:bool=False):
    image, label, rand_int, label_sum = [i.to(self.device) for i in batch]
    pred_label, pred_sum = self.model(image, rand_int)

    if train:
      loss_mnist = self.loss_function['loss_mnist'](pred_label, label)
      loss_sum = self.loss_function['loss_sum'](pred_sum, label_sum.type(torch.long))

      self.optimizer.zero_grad()

      loss_mnist.backward()
      loss_sum.backward()
      self.optimizer.step()

    for metric, metric_obj in self.metrics.items():
      if 'mnist' in metric:
        metric_obj(pred_label.argmax(dim=1), label)
      elif 'sum' in metric:
        metric_obj(pred_sum.argmax(dim=1), label_sum)

    if train:
      return {
        'loss_mnist': loss_mnist.item(),
        'loss_sum': loss_sum.item()
      }


  def _init_logger(self):
    pass

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
