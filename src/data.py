import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import pytorch_lightning as pl

class ImageFolderDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, train_size=0.8, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_size = train_size
        self.transform = transform

    def setup(self, stage=None):
        dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
        train_size = int(self.train_size * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True , num_workers=31)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=31)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
