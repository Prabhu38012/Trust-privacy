"""Dataset for deepfake detection training"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class DeepfakeDataset(Dataset):
    """
    Dataset for deepfake detection.
    
    Label convention:
    - 0 = REAL (authentic)
    - 1 = FAKE (deepfake/manipulated)
    
    Directory structure:
    data/
    ├── real/    → label 0
    └── fake/    → label 1
    """
    
    def __init__(self, data_dir, transform=None, split='train', train_ratio=0.8):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Collect images
        self.samples = []
        
        real_dir = self.data_dir / "real"
        fake_dir = self.data_dir / "fake"
        
        # REAL images → label 0
        if real_dir.exists():
            for img_path in real_dir.glob("*"):
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                    self.samples.append((img_path, 0))  # 0 = REAL
        
        # FAKE images → label 1
        if fake_dir.exists():
            for img_path in fake_dir.glob("*"):
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                    self.samples.append((img_path, 1))  # 1 = FAKE
        
        # Shuffle and split
        import random
        random.seed(42)
        random.shuffle(self.samples)
        
        split_idx = int(len(self.samples) * train_ratio)
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        # Print distribution
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        print(f"  {split.upper()} set: {real_count} real (label=0), {fake_count} fake (label=1)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)


def get_dataloaders(data_dir, batch_size=32, image_size=224, num_workers=0):
    """Create train and validation dataloaders"""
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    print(f"\nLoading dataset from: {data_dir}")
    train_dataset = DeepfakeDataset(data_dir, transform=train_transform, split='train')
    val_dataset = DeepfakeDataset(data_dir, transform=val_transform, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
