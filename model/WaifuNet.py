import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

class WaifuNet(nn.Module):

    def __init__(self, class_num):

        super(WaifuNet, self).__init__()
        pass
    
    def forward(self, x):
        
        return x

if __name__ == '__main__':

    net = WaifuNet()
    print(net)
    rand = torch.randn(2, 4, 240, 320)
    output = net(rand)
    print(output.shape)
    