"""
Document Fraud Detection CNN Model
===================================
A CNN-based classifier for detecting fake/fraudulent documents.
Uses EfficientNet-B0 as backbone with custom classifier head.

Usage:
    python train_document_model.py --data_dir ./document_dataset --epochs 20
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class DocumentFraudCNN(nn.Module):
    """
    CNN for document fraud detection using EfficientNet-B0 backbone.
    
    Classes:
        0 = AUTHENTIC (real document)
        1 = FRAUDULENT (fake/sample/edited document)
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # Use EfficientNet-B0 as backbone
        if pretrained:
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Get number of features from backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def predict_proba(self, x):
        """Get probability scores for each class."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs


# ============================================================
# DATASET
# ============================================================

class DocumentDataset(Dataset):
    """
    Dataset for document fraud detection.
    
    Expected folder structure:
        data_dir/
            real/
                doc1.jpg
                doc2.png
                ...
            fake/
                fake1.jpg
                sample1.png
                ...
    """
    
    def __init__(self, data_dir, transform=None, split='train', train_ratio=0.8):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        self.samples = []
        self.labels = []
        
        # Load real documents (label = 0)
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for img_path in real_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                    self.samples.append(str(img_path))
                    self.labels.append(0)
        
        # Load fake documents (label = 1)
        fake_dir = self.data_dir / 'fake'
        if fake_dir.exists():
            for img_path in fake_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                    self.samples.append(str(img_path))
                    self.labels.append(1)
        
        # Shuffle and split
        indices = np.arange(len(self.samples))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * train_ratio)
        
        if split == 'train':
            indices = indices[:split_idx]
        else:
            indices = indices[split_idx:]
        
        self.samples = [self.samples[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        
        print(f"[{split.upper()}] Loaded {len(self.samples)} samples")
        print(f"  - Real: {self.labels.count(0)}, Fake: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image on error
            return torch.zeros(3, 224, 224), label


# ============================================================
# TRANSFORMS
# ============================================================

def get_transforms(is_training=True):
    """Get image transforms for training/validation."""
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# ============================================================
# TRAINING
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}: Loss={loss.item():.4f}")
    
    return total_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100.0 * correct / total


def train_model(data_dir, output_dir, epochs=20, batch_size=16, lr=0.001):
    """Main training function."""
    
    print("\n" + "="*60)
    print("DOCUMENT FRAUD CNN TRAINING")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create datasets
    train_dataset = DocumentDataset(
        data_dir, 
        transform=get_transforms(is_training=True),
        split='train'
    )
    val_dataset = DocumentDataset(
        data_dir,
        transform=get_transforms(is_training=False),
        split='val'
    )
    
    if len(train_dataset) == 0:
        print("\n❌ ERROR: No training data found!")
        print(f"Expected folder structure:")
        print(f"  {data_dir}/")
        print(f"    real/   (authentic documents)")
        print(f"    fake/   (fake/sample documents)")
        return None
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = DocumentFraudCNN(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path = Path(output_dir) / 'document_fraud_model.pth'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, output_path)
            print(f"✅ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "="*60)
    print(f"TRAINING COMPLETE! Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {output_dir}/document_fraud_model.pth")
    print("="*60)
    
    # Save training history
    history_path = Path(output_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return model


# ============================================================
# INFERENCE
# ============================================================

def load_model(model_path, device='cpu'):
    """Load trained model for inference."""
    model = DocumentFraudCNN(num_classes=2, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def predict_document(model, image_path, device='cpu'):
    """
    Predict if a document is real or fake.
    
    Returns:
        dict with prediction, confidence, and probabilities
    """
    transform = get_transforms(is_training=False)
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        
    prob_real = probs[0][0].item()
    prob_fake = probs[0][1].item()
    
    prediction = "AUTHENTIC" if prob_real > prob_fake else "FRAUDULENT"
    confidence = max(prob_real, prob_fake)
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'prob_authentic': prob_real,
        'prob_fraudulent': prob_fake
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Document Fraud CNN')
    parser.add_argument('--data_dir', type=str, default='./document_dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
