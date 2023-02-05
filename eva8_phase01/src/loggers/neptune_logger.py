#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Neptune logger interface'''

# imports
import os
import neptune.new as neptune
#  script imports
# imports


class NeptuneLogger:
  '''Neptune logger interface'''

  def __init__(self, project_name, project_params):
    self.run = neptune.init_run(
      project=project_name,
      api_token=os.getenv('NEPTUNE_API_TOKEN')
    )

    self.log_param(project_params)

  def log_metric(self, msg, metric_value):
    self.run[msg].append(metric_value)

  def log_param(self, project_params):
    self.run['model/params'] = {k: str(v) for k, v in project_params.items()}

  def stop_neptune(self):
    self.run.stop()
