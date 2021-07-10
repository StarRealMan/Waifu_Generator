import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

class myDataset():
    def __init__(self, dataroot, image_size, workers, batchsize):
        self.dataset = dset.ImageFolder(root=dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

        self.datasetFlip = dset.ImageFolder(root=dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    transforms.RandomHorizontalFlip(1)
                                ]))
        
        assert self.dataset
        assert self.datasetFlip
        self.dataloader = torch.utils.data.DataLoader(self.dataset + self.datasetFlip, batch_size=batchsize,
                                                shuffle=True, num_workers=int(workers))

    def get_dataloader(self):
        return self.dataloader


if __name__ == "__main__":
    img_size = 256
    myDataset = myDataset("~/Pytorch_DEV/Waifu_Generator/data/waifu_2x", img_size, 8, 1)
    myDataloader = myDataset.get_dataloader()
    for i, data in enumerate(myDataloader, 0):
        print(data[0].shape)
        vutils.save_image(data[0], 'dataset.png', normalize=True)
        break