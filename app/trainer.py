import sys
from torch.utils.data import dataset
sys.path.append("..")
from model import WaifuNet
from app import dataloader
import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=8, help='input batch size')
parser.add_argument('--nepoch', type=int, default=8, help='number of epochs to train for')
parser.add_argument('--dataset', type=str, default='/home/starydy/Pytorch_DEV/Waifu_Generator/data/waifu_2x', help='dataset path')
parser.add_argument('--outn', type=str, default='/home/starydy/Pytorch_DEV/Waifu_Generator/trained_model', help='output model name')
parser.add_argument('--model', type=str, default='None', help='history model path')
parser.add_argument('--workers', type=int, default=2, help='number of workers to load data')

opt = parser.parse_args()
print(opt)

train_dataset = dataloader.Waifu2X_Dataset(opt.dataset, 100)
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=opt.batchsize,
                                               num_workers=opt.workers, drop_last=True)

manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
fixed_noise = torch.randn(64, 100, 1, 1, device = device)

real_label = 1.
fake_label = 0.

DisNet = WaifuNet.WaifuNet_Discriminator()
DisNet.to(device)
GenNet = WaifuNet.WaifuNet_Generator()
GenNet.to(device)

optimizerD = optim.Adam(DisNet.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(GenNet.parameters(), lr = 0.0002, betas = (0.5, 0.999))

criterion = nn.BCELoss()
epoch_size = opt.nepoch

for epoch in range(epoch_size):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        
        batch_size = data[0].size
        DisNet.zero_grad()
        real_data = data.to(device)
        label = torch.full((batch_size,), real_label, dtype = torch.float, device = device)
        output = DisNet(real_data).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, 100, 1, 1, device = device)
        fake = GenNet(noise)
        label.fill_(fake_label)
        output = DisNet(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        optimizerD.step()

        GenNet.zero_grad()
        label.fill_(real_label)
        output = DisNet(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()