"""Dataset for deepfake detection training"""

import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np

class DeepfakeDataset(Dataset):
    """Dataset for training deepfake detector"""
    
    def __init__(self, data_dir, split='train', transform=None, face_crop=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.face_crop = face_crop
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """Load real and fake image paths"""
        # Real images
        real_dir = self.data_dir / "real"
        if real_dir.exists():
            for img_path in real_dir.glob("**/*.jpg"):
                self.samples.append((str(img_path), 0))  # 0 = real
            for img_path in real_dir.glob("**/*.png"):
                self.samples.append((str(img_path), 0))
        
        # Fake images
        fake_dir = self.data_dir / "fake"
        if fake_dir.exists():
            for img_path in fake_dir.glob("**/*.jpg"):
                self.samples.append((str(img_path), 1))  # 1 = fake
            for img_path in fake_dir.glob("**/*.png"):
                self.samples.append((str(img_path), 1))
        
        # Shuffle
        random.shuffle(self.samples)
        
        # Split
        split_idx = int(len(self.samples) * 0.8)
        if self.split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        print(f"[{self.split}] Loaded {len(self.samples)} samples")
    
    def _extract_face(self, img):
        """Extract face from image"""
        img_array = np.array(img)
        if len(img_array.shape) == 2:
            gray = img_array
        else:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            margin = int(w * 0.3)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img.width, x + w + margin)
            y2 = min(img.height, y + h + margin)
            return img.crop((x1, y1, x2, y2))
        
        # Center crop fallback
        w, h = img.size
        m = min(w, h)
        return img.crop(((w-m)//2, (h-m)//2, (w+m)//2, (h+m)//2))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            if self.face_crop:
                img = self._extract_face(img)
            
            if self.transform:
                img = self.transform(img)
            
            return img, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random valid sample
            return self.__getitem__(random.randint(0, len(self.samples) - 1))


def get_transforms(image_size=224, is_train=True):
    """Get data transforms"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def get_dataloaders(data_dir, batch_size=32, image_size=224, num_workers=4):
    """Create train and validation dataloaders"""
    train_transform = get_transforms(image_size, is_train=True)
    val_transform = get_transforms(image_size, is_train=False)
    
    train_dataset = DeepfakeDataset(data_dir, split='train', transform=train_transform)
    val_dataset = DeepfakeDataset(data_dir, split='val', transform=val_transform)
    
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
