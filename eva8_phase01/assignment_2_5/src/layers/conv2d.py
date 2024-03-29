#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Custom conv2D implementation with activation and initializer
'''


#imports
import torch
import torch.nn as nn

#   script imports
#imports


# classes
class Conv2D(nn.Module):
  '''
  Custom conv2D implementation with activation and initializer
  '''
  def __init__(self,
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode='zeros',
    device=None,
    dtype=None,
    activation='relu',
    kernel_initializer='xavier_uniform',
  ):
    super().__init__()

    self.conv = nn.Conv2d(in_channels,
      out_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      dilation=dilation,
      groups=groups,
      bias=bias,
      padding_mode=padding_mode,
      device=device,
      dtype=dtype
    )
    self.init_weights(kernel_initializer)

    self.activation = self.get_activation(activation)


  def get_activation(self, activation):
    act_dict = {
      'relu': nn.ReLU(),
      'tanh': nn.Tanh(),
      'sigmoid': nn.Sigmoid()
    }

    return act_dict.get(activation.lower(), nn.ReLU())


  def init_weights(self, kernel_initializer):
    if kernel_initializer.lower() == 'xavier_uniform':
      nn.init.xavier_uniform_(self.conv.weight)
    elif kernel_initializer.lower() == 'kaiming_uniform':
      nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')

    # self.conv.bias.data.fill_(0)


  def forward(self, input):
    return self.activation(self.conv(input))
# classes

def main():
  conv2d = Conv2D(3, 3, kernel_size=3)
  print(conv2d(torch.rand([3, 10, 10])))

if __name__ == '__main__':
  main()
