import sys
from numpy.testing._private.utils import nulp_diff
import os
import numpy as np
import torch
import torch.utils.data as data
import cv2
import argparse
from tqdm import tqdm

class Waifu2X_Dataset(data.Dataset):
    
    def __init__(self, root_d, train_num):

        self.dataset_size = 21551
        self.img_list = []
        for img_num in tqdm(range(min(train_num, self.dataset_size))):
            img = cv2.imread(root_d + "/" + str(img_num + 1) + ".png")
            img = img.transpose((2, 0, 1))
            self.img_list.append(img)

    def __getitem__(self, index):
        
        return self.img_list[index]


    def __len__(self):
        
        return len(self.img_list)

if __name__ == '__main__':

    dataset = Waifu2X_Dataset('/home/starydy/Pytorch_DEV/Waifu_Generator/data/waifu_2x', 100)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=2,
                                             num_workers=8, drop_last=True)
    for i, data in enumerate(dataloader):
        img = data
        print(img)
        print(img.shape)
        break
