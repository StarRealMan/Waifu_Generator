import sys

from numpy.testing._private.utils import nulp_diff
import os
import numpy as np
import torch
import torch.utils.data as data
import cv2
import argparse

class Waifu2X_Dataset(data.Dataset):

    def __init__(self, root_d):
        pass

    def __getitem__(self, index):
        pass


    def __len__(self):
        pass

if __name__ == '__main__':

    dataset = Waifu2X_Dataset('../data/SUNRGBD', 'train')
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=2,\
                                             num_workers=8, drop_last=True)
    for i, data in enumerate(dataloader):
        rgb,depth,label = data
        print(rgb)
        print(depth)
        print(label)
        print(label.shape)
        break;
