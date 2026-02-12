import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import config

class DFC_Dataset(Dataset):
    def __init__(self, images, labels=None, use_8bit=True, is_train=True):

        self.images = images
        self.labels = labels
        self.use_8bit = use_8bit
        self.is_train = is_train
        
    def __len__(self):
        return len(self.images)
    
    def normalize(self, x):
        x = x.float()
        imgs = []
        for channel in range(x.shape[0]):
            min_value = x[channel].mean() - 2 * x[channel].std()
            max_value = x[channel].mean() + 2 * x[channel].std()
            
            if self.use_8bit:
                img = (x[channel] - min_value) / (max_value - min_value) * 255.0
                img = torch.clip(img, 0, 255).to(torch.uint8)
            else:
                img = (x[channel] - min_value) / (max_value - min_value)
                img = torch.clip(img, 0, 1)
            imgs.append(img)
        
        return torch.stack(imgs, dim=0)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.normalize(image)
        
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        return image


def create_data_loaders(train_images, train_labels, val_images, val_labels, batch_size=64):
    train_dataset = DFC_Dataset(train_images, None, use_8bit=config.USE_8BIT, is_train=True)
    val_dataset = DFC_Dataset(val_images, None, use_8bit=config.USE_8BIT, is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def create_labeled_data_loaders(train_images, train_labels, val_images, val_labels, batch_size=64):
    """Create data loaders with labels for evaluation"""
    train_dataset = DFC_Dataset(train_images, train_labels, use_8bit=config.USE_8BIT, is_train=True)
    val_dataset = DFC_Dataset(val_images, val_labels, use_8bit=config.USE_8BIT, is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def load_dfc_data(path):

    dfc = torch.load(path)
    
    train_images = dfc["train_images"]  # (46_152, 14, 96, 96)
    train_labels = dfc["train_labels"]  # (46_152, 96, 96)
    validation_images = dfc["validation_images"]  # (8_874, 14, 96, 96)
    validation_labels = dfc["validation_labels"]  # (8_874, 96, 96)
    
    return train_images, train_labels, validation_images, validation_labels