"""Training script for deepfake detection model"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from config import CONFIG
from dataset import get_dataloaders
from model import get_model


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs.view(-1), targets.view(-1))
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.1f}%'
        })
    
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    
    # Calculate AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.0
    
    return total_loss / len(loader), accuracy, auc


def train(args):
    """Main training function"""
    print("=" * 60)
    print("TrustLock - Deepfake Detection Model Training")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create directories
    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    models_dir = Path(CONFIG["models_dir"])
    checkpoint_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    # Data
    data_dir = args.data_dir or CONFIG["data_dir"]
    print(f"\nLoading data from: {data_dir}")
    
    # Check if data exists
    real_dir = Path(data_dir) / "real"
    fake_dir = Path(data_dir) / "fake"
    
    real_count = len(list(real_dir.glob("*.jpg"))) + len(list(real_dir.glob("*.png"))) if real_dir.exists() else 0
    fake_count = len(list(fake_dir.glob("*.jpg"))) + len(list(fake_dir.glob("*.png"))) if fake_dir.exists() else 0
    
    if real_count == 0 or fake_count == 0:
        print("\n❌ No training data found!")
        print(f"   Real images: {real_count}")
        print(f"   Fake images: {fake_count}")
        print("\n📝 Run this first: python download_data.py")
        print("   Or add images manually to data/real/ and data/fake/")
        return
    
    print(f"   Found {real_count} real + {fake_count} fake images")
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(
        data_dir,
        batch_size=CONFIG["batch_size"],
        image_size=CONFIG["image_size"],
        num_workers=0  # Use 0 for Windows compatibility
    )
    
    # Model
    print("\nInitializing model...")
    model = get_model(
        encoder=CONFIG["model"],
        pretrained=True,
        device=device
    )
    
    # Training setup
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    
    # Training loop
    best_auc = 0
    best_acc = 0
    
    print(f"\nStarting training for {CONFIG['epochs']} epochs...")
    print("-" * 60)
    
    for epoch in range(CONFIG["hi"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_auc = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_acc = val_acc
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"✅ Saved best model (AUC: {val_auc:.4f})")
        
        # Save latest
        torch.save(model.state_dict(), checkpoint_dir / "latest_model.pth")
    
    # Save final model to models directory
    final_path = models_dir / "deepfake_detector.pth"
    torch.save(model.state_dict(), final_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_acc*100:.1f}%")
    print(f"Best Validation AUC: {best_auc:.4f}")
    print(f"Model saved to: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deepfake detector")
    parser.add_argument("--data_dir", type=str, help="Path to training data")
    args = parser.parse_args()
    train(args)
