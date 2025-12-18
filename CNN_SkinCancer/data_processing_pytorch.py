# data_processing_pytorch.py
# This module handles loading and preprocessing of the Skin Cancer dataset using PyTorch

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

class SkinCancerDataset(Dataset):
    """PyTorch Dataset for Skin Cancer images"""
    
    def __init__(self, dataframe, transform=None):
        """
        Arguments:
        - dataframe: pandas DataFrame with 'filepath' and 'label' columns
        - transform: torchvision transforms to apply to images
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        
        # Create label to index mapping
        self.classes = sorted(self.dataframe['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        img_path = self.dataframe.iloc[idx]['filepath']
        label_name = self.dataframe.iloc[idx]['label']
        label_idx = self.class_to_idx[label_name]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx

def load_skin_cancer_data(df_train, df_test, img_size=(64, 64), batch_size=32, validation_split=0.1):
    """
    Load skin cancer images using PyTorch DataLoaders.
    
    Arguments:
    - df_train: pandas DataFrame with training data
    - df_test: pandas DataFrame with test data
    - img_size: tuple, target image size (height, width)
    - batch_size: int, batch size for dataloaders
    - validation_split: float, fraction of training data to use for validation
    
    Returns:
    - train_loader: DataLoader for training
    - val_loader: DataLoader for validation
    - test_loader: DataLoader for testing
    - class_labels: list of class names
    """
    
    # Split training data into train and validation
    df_train_shuffled = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    val_size = int(len(df_train_shuffled) * validation_split)
    
    df_train_split = df_train_shuffled[val_size:]
    df_val_split = df_train_shuffled[:val_size]
    
    # Define transforms
    # Training: data augmentation
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/Test: no augmentation
    val_test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SkinCancerDataset(df_train_split, transform=train_transform)
    val_dataset = SkinCancerDataset(df_val_split, transform=val_test_transform)
    test_dataset = SkinCancerDataset(df_test, transform=val_test_transform)
    
    # Create dataloaders (num_workers=0 for macOS compatibility)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Get class labels
    class_labels = train_dataset.classes
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes ({len(class_labels)}): {class_labels}")
    
    return train_loader, val_loader, test_loader, class_labels
