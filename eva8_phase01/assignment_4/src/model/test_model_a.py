#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
TestModel for Assignment 3
'''


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
  def __init__(self, in_channels: int, out_channels: int, dropout: float):

    super(ConvBlock, self).__init__()
    self.conv_1 = Conv2D(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=(3, 3),
      padding=1
    )
    self.batch_norm_1 = nn.BatchNorm2d(num_features=out_channels, affine=True)
    self.relu_1 = nn.ReLU(inplace=False)

    self.conv_2 = Conv2D(
      in_channels=out_channels,
      out_channels=out_channels,
      kernel_size=(3, 3),
      padding=1
    )
    self.batch_norm_2 = nn.BatchNorm2d(num_features=out_channels, affine=True)
    self.relu_2 = nn.ReLU(inplace=False)

    self.pool_1 = nn.AvgPool2d(kernel_size=(2, 2))

    self.dropout_1 = nn.Dropout(p=dropout)

  def forward(self, input: torch.Tensor):
    x = input
    x = self.conv_1(x)
    x = self.batch_norm_1(x)
    x = self.relu_1(x)
    x = self.conv_2(x)
    x = self.batch_norm_2(x)
    x = self.relu_2(x)
    x = self.pool_1(x)
    x = self.dropout_1(x)
    return x



class TestModelA(nn.Module):
  '''
  docstring for TestModelA
  '''

  def __init__(self, num_classes: int):
    super(TestModelA, self).__init__()

    # LAYERS FOR IMAGE RECOGNITION
    self.conv_block_1 = ConvBlock(in_channels= 1, out_channels=16, dropout=0.20)
    self.conv_block_2 = ConvBlock(in_channels=16, out_channels=16, dropout=0.25)
    self.conv_block_3 = ConvBlock(in_channels=16, out_channels=16, dropout=0.20)

    self.gap_layer = Conv2D(in_channels=16, out_channels=num_classes, kernel_size=(1, 1))

    self.dense = Dense(in_features=90, out_features=num_classes, activation='log_softmax')

  def forward(self, input: torch.Tensor):
    x = input

    x = self.conv_block_1(x)
    x = self.conv_block_2(x)
    x = self.conv_block_3(x)

    x = self.gap_layer(x)
    x = x.view(x.size(0), -1)
    x = self.dense(x)

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
