#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Train the model'''

#imports
from argparse import ArgumentParser

#   script imports
from engine.lenet_engine import LeNetEngine
from utils.module_params_parser import ModelParamsDecoder

#imports

def main(hparams=None):
  pe = LeNetEngine(hparams)
  pe.train()
  pe.test()


if __name__ == '__main__':
  parser = ArgumentParser(parents=[])

  parser.add_argument('--params_yml', type=str)
  parser.add_argument('--kernel_platform', type=str, default='colab')

  arg_params = parser.parse_args()

  hyperparams = ModelParamsDecoder(
    arg_params.params_yml,
    arg_params.kernel_platform
  )

  main(hparams=hyperparams)
