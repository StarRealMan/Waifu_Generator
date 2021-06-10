import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import cv2

class WaifuNet_Generator(nn.Module):

    def __init__(self):

        super(WaifuNet_Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),  # 100 x 1 x 1
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 512 x 4 x 4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),  # 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 256 x 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),  # 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),   # 128 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),     # 64 x 128 x 128
            nn.Tanh()                                           # 3 x 256 x 256
        )
    
    def forward(self, x):
        
        return self.main(x)

class WaifuNet_Discriminator(nn.Module):

    def __init__(self):

        super(WaifuNet_Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),      # 3 x 256 x 256
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),    # 64 x 128 x 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1, bias=False),   # 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),   # 128 x 32 x 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 4, 2, 1, bias=False),   # 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),   # 256 x 8 x 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),     # 512 x 4 x 4
            nn.Sigmoid()                                # 1 x 256 x 256
        )
    
    def forward(self, x):
        
        return self.main(x)

if __name__ == '__main__':

    Dnet = WaifuNet_Discriminator()
    print(Dnet)
    rand = torch.randn(2, 3, 256, 256)
    output = Dnet(rand)
    print(output.shape)

    Gnet = WaifuNet_Generator()
    print(Gnet)
    rand = torch.randn(2, 100, 1, 1)
    output = Gnet(rand).detach().numpy()
    
    # cv2.imshow("gen", output[1].transpose((1, 2, 0)))
    # cv2.waitKey(0)
    # print(output.shape)
    