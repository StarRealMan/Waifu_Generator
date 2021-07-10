from __future__ import print_function
import argparse
import sys
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
sys.path.append("..")
from model import WaifuNet
from model import originDCGAN

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--netG', default='../model/netG_64.pth', help="path to netG (to infer)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3

device = torch.device("cuda:0" if opt.cuda else "cpu")

netG = WaifuNet.Generator(ngpu, nz, nc, ngf).to(device)
# netG = originDCGAN.Generator(ngpu, nz, nc, ngf).to(device)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

with torch.no_grad():
    netG.eval()
    latent_z = torch.randn(opt.batchSize, nz, 1, 1, device=device)
    inference = netG(latent_z)

    vutils.save_image(inference,
            '%s/fake_samples_infernece.png' % (opt.outf),
            normalize=True)