#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''neptune Callback interface'''

# imports
#  script imports
from loggers.neptune_logger import NeptuneLogger
from callbacks.callback_handler import Callback
# imports


class NeptuneCallback(Callback):
  def __init__(self, project_name, project_params):
    self.logger = NeptuneLogger(
      project_name=project_name,
      project_params=project_params
    )

  def on_train_end(self, **kwargs):
    self.logger.stop_neptune()

  def on_epoch_end(self, **kwargs):
    self.logger.log_metric(
      msg='metrics/epoch/accuracy',
      metric_value=kwargs['m_mnist_accuracy']
    )

    self.logger.log_metric(
      msg='metrics/epoch/loss',
      metric_value=kwargs['loss_mnist']
    )

  def on_batch_end(self, **kwargs):
    self.logger.log_metric(
      msg='metrics/batch/accuracy',
      metric_value=kwargs['m_mnist_accuracy']
    )

    self.logger.log_metric(
      msg='metrics/batch/loss',
      metric_value=kwargs['loss_mnist']
    )

  def on_test_batch_end(self, **kwargs):
    self.logger.log_metric(
      msg='test/metrics/batch/accuracy',
      metric_value=kwargs['m_mnist_accuracy']
    )

    self.logger.log_metric(
      msg='test/metrics/batch/loss',
      metric_value=kwargs['loss_mnist']
    )

  def on_test_epoch_end(self, **kwargs):
    self.logger.log_metric(
      msg='test/metrics/epoch/accuracy',
      metric_value=kwargs['test_m_mnist_accuracy']
    )

    self.logger.log_metric(
      msg='test/metrics/epoch/loss',
      metric_value=kwargs['test_loss_mnist']
    )

  def run_predictions(self, **kwargs):
    self.logger.log_predictions(
      msg='test/predictions',
      image=kwargs['image'],
      label=kwargs['label'],
      pred_label=kwargs['pred_label']
    )
