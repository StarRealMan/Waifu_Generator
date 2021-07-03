import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

class myDataset():
    def __init__(self, dataroot, image_size, workers, batchsize):
        self.dataset = dset.ImageFolder(root=dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        
        assert self.dataset
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batchsize,
                                                shuffle=True, num_workers=int(workers))

    def get_dataloader(self):
        return self.dataloader

