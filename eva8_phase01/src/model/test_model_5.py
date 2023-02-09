#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''TestModel for Assignment 5'''


#imports
import torch
import torch.nn as nn
from typing import Any
#   script imports
from layers.conv2d import Conv2D
from layers.dense import Dense

#imports


# classes

class ConvBlock(nn.Module):
  '''Creating the Convolution block'''
  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      dropout:float=0.1,
      padding:int=0,
      norm_type:str='batch',
      layer_params:Any=4
    ):

    super(ConvBlock, self).__init__()
    self.conv_1 = Conv2D(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=(3, 3),
      padding=padding,
      bias=False
    )

    # adding normalization layer
    if norm_type == 'batch':
      self.norm_1 = nn.BatchNorm2d(num_features=out_channels)
    elif norm_type == 'layer':
      self.norm_1 = nn.LayerNorm(normalized_shape=[out_channels, *layer_params])
    elif norm_type == 'group':
      self.norm_1 = nn.GroupNorm(num_groups=layer_params, num_channels=out_channels)
    # adding normalization layer

    self.relu_1 = nn.ReLU(inplace=False)

    self.dropout_1 = nn.Dropout(p=dropout)

  def forward(self, input: torch.Tensor):
    x = input
    x = self.conv_1(x)
    x = self.norm_1(x)
    x = self.relu_1(x)
    x = self.dropout_1(x)
    return x

class TransitionBlock(nn.Module):
  '''Creating the Transition block'''
  def __init__(self, in_channels: int, out_channels: int, padding:int=0):
    super(TransitionBlock, self).__init__()
    self.conv_1 = Conv2D(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=(1, 1),
      padding=padding,
      bias=False
    )

    # self.batch_norm_1 = nn.BatchNorm2d(num_features=out_channels)
    # self.relu_1 = nn.ReLU(inplace=False)

  def forward(self, input:torch.Tensor):
    x = input
    x = self.conv_1(x)
    # x = self.batch_norm_1(x)
    # x = self.relu_1(x)

    return x


class TestModel5(nn.Module):
  '''TestModel for Assignment 5'''

  def __init__(self, num_classes: int, norm_type:str='batch'):
    super(TestModel5, self).__init__()

    if norm_type == 'batch':
      groups = 4
    elif norm_type == 'group':
      groups = 4

    # LAYERS FOR IMAGE RECOGNITION
    if norm_type == 'layer': groups = [26, 26]
    self.conv_block_1 = ConvBlock(in_channels= 1, out_channels=16, norm_type=norm_type, layer_params=groups)

    if norm_type == 'layer': groups = [24, 24]
    self.conv_block_2 = ConvBlock(in_channels=16, out_channels=16, norm_type=norm_type, layer_params=groups)

    self.trans_block_1 = TransitionBlock(in_channels=16, out_channels=10)

    self.pool_1 = nn.MaxPool2d(2, 2)

    if norm_type == 'layer': groups = [10, 10]
    self.conv_block_3 = ConvBlock(in_channels=10, out_channels=16, norm_type=norm_type, layer_params=groups)

    if norm_type == 'layer': groups = [8, 8]
    self.conv_block_4 = ConvBlock(in_channels=16, out_channels=16, norm_type=norm_type, layer_params=groups)

    groups = [6, 6] if norm_type == 'layer' else 2
    self.conv_block_5 = ConvBlock(in_channels=16, out_channels=10, norm_type=norm_type, layer_params=groups)

    groups = [6, 6] if norm_type == 'layer' else 2
    self.conv_block_6 = ConvBlock(in_channels=10, out_channels=10, padding=1, norm_type=norm_type, layer_params=groups)

    self.conv_1 = Conv2D(
      in_channels=10,
      out_channels=10,
      kernel_size=(1, 1),
      bias=False
    )

    self.gap_layer_1 = nn.AvgPool2d(kernel_size=(6, 6))

    self.log_softmax = nn.LogSoftmax(dim=-1)

  def forward(self, input: torch.Tensor):
    x = input

    x = self.conv_block_1(x)
    x = self.conv_block_2(x)

    x = self.trans_block_1(x)

    x = self.pool_1(x)

    x = self.conv_block_3(x)
    x = self.conv_block_4(x)
    x = self.conv_block_5(x)
    x = self.conv_block_6(x)

    x = self.gap_layer_1(x)

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
