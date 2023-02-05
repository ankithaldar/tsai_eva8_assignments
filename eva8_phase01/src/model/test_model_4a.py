#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''TestModel for Assignment 4'''


#imports
import torch
import torch.nn as nn
#   script imports
from layers.conv2d import Conv2D
from layers.dense import Dense

#imports


# classes

class ConvBlock(nn.Module):
  '''Creating the Convolution block'''
  def __init__(self, in_channels: int, out_channels: int, padding:int=0, pool:bool=False):

    super(ConvBlock, self).__init__()
    self.conv_1 = Conv2D(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=(3, 3),
      padding=padding,
      bias=False
    )
    self.batch_norm_1 = nn.BatchNorm2d(num_features=out_channels)
    self.relu_1 = nn.ReLU(inplace=False)

    self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))

    self.if_pool = pool

  def forward(self, input: torch.Tensor):
    x = input
    x = self.conv_1(x)
    x = self.batch_norm_1(x)
    x = self.relu_1(x)
    if self.if_pool:
      x = self.pool_1(x)
    # x = self.dropout_1(x)
    return x

class TransitionBlock(nn.Module):
  '''Creating the Transition block'''
  def __init__(self, in_channels: int, out_channels: int, padding:int=0, pool:bool=False):
    super(TransitionBlock, self).__init__()
    self.conv_1 = Conv2D(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=(1, 1),
      padding=padding,
      bias=False
    )

    self.batch_norm_1 = nn.BatchNorm2d(num_features=out_channels)
    self.relu_1 = nn.ReLU(inplace=False)

  def forward(self, input:torch.Tensor):
    x = input
    x = self.conv_1(x)
    x = self.batch_norm_1(x)
    x = self.relu_1(x)

    return x


class TestModel4A(nn.Module):
  '''TestModel for Assignment 4'''

  def __init__(self, num_classes: int):
    super(TestModel4A, self).__init__()

    # LAYERS FOR IMAGE RECOGNITION
    self.conv_block_1 = ConvBlock(in_channels= 1, out_channels=10)
    self.conv_block_2 = ConvBlock(in_channels=10, out_channels=10)
    self.conv_block_3 = ConvBlock(in_channels=10, out_channels=20, pool=True)

    self.trans_block_1 = TransitionBlock(in_channels=20, out_channels=10)

    self.conv_block_4 = ConvBlock(in_channels=10, out_channels=10)
    self.conv_block_5 = ConvBlock(in_channels=10, out_channels=10)
    self.trans_block_2 = TransitionBlock(in_channels=10, out_channels=10)

    self.conv_1 = Conv2D(
      in_channels=10,
      out_channels=10,
      kernel_size=(7, 7),
      bias=False
    )

    self.log_softmax = nn.LogSoftmax(dim=-1)

  def forward(self, input: torch.Tensor):
    x = input

    x = self.conv_block_1(x)
    x = self.conv_block_2(x)
    x = self.conv_block_3(x)
    x = self.trans_block_1(x)

    x = self.conv_block_4(x)
    x = self.conv_block_5(x)
    x = self.trans_block_2(x)

    x = self.conv_1(x)
    x = x.view(-1, 10)
    x = self.log_softmax(x)

    return x
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
