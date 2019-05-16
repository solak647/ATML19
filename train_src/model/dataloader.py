from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, Grayscale, ColorJitter
from torch.utils import data

class DataLoader:
    
    def __init__(self, data_folder, train_transforms = None, test_transforms = None, val_transforms = None, batch_size = 100):
        self.data_folder = data_folder
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.val_transforms = val_transforms
        self.batch_size = batch_size
        if self.train_transforms is None:
            self.train_transforms = Compose([
                ColorJitter(brightness=0.2,contrast=0.2),
                ToTensor(),           # Converts to Tensor, scales to [0, 1] float (from [0, 255] int)
                Normalize((0.5,), (0.5,)), # scales to [-1.0, 1.0]
            ])
        if self.test_transforms is None:
            self.test_transforms = Compose([
                ToTensor(),           # Converts to Tensor, scales to [0, 1] float (from [0, 255] int)
                Normalize((0.5,), (0.5,)), # scales to [-1.0, 1.0]
            ])
        if self.val_transforms is None:
            self.val_transforms = Compose([
                ToTensor(),           # Converts to Tensor, scales to [0, 1] float (from [0, 255] int)
                Normalize((0.5,), (0.5,)), # scales to [-1.0, 1.0]
            ])
        self.train_dir = self.data_folder + '/train'
        self.val_dir = self.data_folder + '/val'
        self.test_dir = self.data_folder + '/test'
        pass
    
    def get_dataloader(self):
        train_dataset = ImageFolder(self.train_dir, transform=self.train_transforms)
        train_dataloader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=32)
        val_dataset = ImageFolder(self.val_dir, transform=self.val_transforms)
        val_dataloader = data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        test_dataset = ImageFolder(self.test_dir, transform=self.test_transforms)
        test_dataloader = data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        return train_dataloader, val_dataloader, test_dataloader