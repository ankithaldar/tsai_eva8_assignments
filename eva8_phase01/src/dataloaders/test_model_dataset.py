#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Dataloader for downloading and batching data'''


#imports
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from typing import List
#   script imports
#imports


# classes
class MNISTDataLoader(Dataset):
  '''Dataloader for downloading and batching data'''

  def __init__(self, root:str='', train=False, augments='None'):
    # super(DataLoader, self).__init__()
    self.augments = augments
    self.dataset = self.__download_dataset__(root=root, train=train)

    self.labels = self.dataset.targets
    self.image = self.dataset.data

  def __download_dataset__(self, root:str='', train=False):
    '''Downloads the dataset'''

    return torchvision.datasets.MNIST(
      root=root,
      train=train,
      download=True,
      transform=transforms.Compose(self.__get_augments__(train))
    )

  def __len__(self):
    return len(self.image)

  def __getitem__(self, index:int):
    '''Gets the sample'''

    image = self.image[index]
    label = self.labels[index]

    return (image/1.).reshape([1, 28, 28]), label

  def __get_augments__(self, train=False):
    augments = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    if not train:
      return augments

    if isinstance(self.augments, list) and train:
      for each in self.augments:
        if each == 'RandomRotation':
          augments.insert(0, transforms.RandomRotation((-7.0, 7.0), fill=(1,)))
        elif each == 'ColorJitter':
          augments.insert(0, transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1))

      return augments

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
