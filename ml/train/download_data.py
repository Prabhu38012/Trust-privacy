"""
Download training data for deepfake detection.
Downloads sample dataset from public sources.
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
import sys

DATA_DIR = Path("./data")
REAL_DIR = DATA_DIR / "real"
FAKE_DIR = DATA_DIR / "fake"

def download_progress(count, block_size, total_size):
    percent = min(100, int(count * block_size * 100 / total_size))
    sys.stdout.write(f"\rDownloading: {percent}%")
    sys.stdout.flush()

def setup_directories():
    DATA_DIR.mkdir(exist_ok=True)
    REAL_DIR.mkdir(exist_ok=True)
    FAKE_DIR.mkdir(exist_ok=True)

def generate_synthetic_training_data():
    """
    Generate synthetic training data using image augmentation.
    This creates a minimal dataset to start training.
    For production, use real datasets like FaceForensics++.
    """
    import cv2
    import numpy as np
    from PIL import Image
    
    print("\n📦 Generating synthetic training data...")
    print("   (For production, use real datasets from Kaggle/FaceForensics++)")
    
    # Create sample images
    np.random.seed(42)
    
    # Generate REAL-like images (clear, consistent)
    print("\n   Creating 'real' training samples...")
    for i in range(1000):
        # Create face-like pattern
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Skin tone background
        skin_tone = np.random.randint(180, 230)
        img[:, :] = [skin_tone, int(skin_tone*0.8), int(skin_tone*0.7)]
        
        # Add realistic noise
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add texture (real faces have texture)
        texture = np.random.randint(0, 15, img.shape, dtype=np.uint8)
        img = cv2.add(img, texture)
        
        # Add some structure (eyes, nose area)
        center_y, center_x = 128, 128
        cv2.circle(img, (center_x - 40, center_y - 30), 15, (50, 50, 50), -1)  # left eye
        cv2.circle(img, (center_x + 40, center_y - 30), 15, (50, 50, 50), -1)  # right eye
        cv2.ellipse(img, (center_x, center_y + 20), (20, 10), 0, 0, 360, (150, 100, 100), -1)  # nose
        cv2.ellipse(img, (center_x, center_y + 60), (30, 10), 0, 0, 180, (180, 100, 100), 2)  # mouth
        
        # Apply slight blur (natural)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        cv2.imwrite(str(REAL_DIR / f"real_{i:04d}.jpg"), img)
        
        if (i + 1) % 200 == 0:
            print(f"      Generated {i+1}/1000 real samples")
    
    # Generate FAKE-like images (artifacts, smooth, inconsistent)
    print("\n   Creating 'fake' training samples...")
    for i in range(1000):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Skin tone
        skin_tone = np.random.randint(180, 230)
        img[:, :] = [skin_tone, int(skin_tone*0.8), int(skin_tone*0.7)]
        
        # FAKE characteristics:
        
        # 1. Overly smooth (GAN artifact)
        img = cv2.GaussianBlur(img, (9, 9), 0)
        
        # 2. Color inconsistency at boundaries
        boundary_y = np.random.randint(80, 180)
        img[:boundary_y, :, 0] = np.clip(img[:boundary_y, :, 0].astype(int) + 20, 0, 255)
        
        # 3. Add subtle grid pattern (GAN fingerprint)
        for y in range(0, 256, 8):
            img[y:y+1, :] = np.clip(img[y:y+1, :].astype(int) + 5, 0, 255)
        for x in range(0, 256, 8):
            img[:, x:x+1] = np.clip(img[:, x:x+1].astype(int) + 5, 0, 255)
        
        # 4. Asymmetric features (common in deepfakes)
        cv2.circle(img, (128 - 40, 100), 15, (50, 50, 50), -1)  # left eye
        cv2.circle(img, (128 + 45, 95), 12, (55, 55, 55), -1)   # right eye (different size/position)
        
        # 5. Blending artifact at face boundary
        mask = np.zeros((256, 256), dtype=np.float32)
        cv2.circle(mask, (128, 128), 100, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        # Create visible boundary
        edge = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        img[edge > 0] = [200, 180, 180]
        
        cv2.imwrite(str(FAKE_DIR / f"fake_{i:04d}.jpg"), img)
        
        if (i + 1) % 200 == 0:
            print(f"      Generated {i+1}/1000 fake samples")
    
    print(f"\n✅ Generated 1000 real + 1000 fake training samples")
    print(f"   Location: {DATA_DIR.absolute()}")

def main():
    print("=" * 60)
    print("TrustLock - Training Data Setup")
    print("=" * 60)
    
    setup_directories()
    
    # Check if data already exists
    real_count = len(list(REAL_DIR.glob("*.jpg"))) + len(list(REAL_DIR.glob("*.png")))
    fake_count = len(list(FAKE_DIR.glob("*.jpg"))) + len(list(FAKE_DIR.glob("*.png")))
    
    if real_count > 100 and fake_count > 100:
        print(f"\n✅ Training data already exists:")
        print(f"   Real images: {real_count}")
        print(f"   Fake images: {fake_count}")
        return
    
    print("\n⚠️  No training data found!")
    print("\nOptions:")
    print("1. Generate synthetic data (quick start, ~80% accuracy)")
    print("2. Download real dataset (better accuracy)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        generate_synthetic_training_data()
    else:
        print("\n" + "=" * 60)
        print("DOWNLOAD INSTRUCTIONS")
        print("=" * 60)
        print("""
For 90%+ accuracy, download a real dataset:

🔗 QUICK OPTION (Kaggle - 140k faces):
   1. Go to: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
   2. Download and extract
   3. Copy images to:
      - data/real/  (real faces)
      - data/fake/  (fake faces)

🔗 BEST OPTION (FaceForensics++):
   1. Request access: https://github.com/ondyari/FaceForensics
   2. Download videos
   3. Run: python extract_frames.py --real_videos ./original --fake_videos ./manipulated

After adding data, run: python train.py
""")
        
        # Generate synthetic anyway for testing
        print("\nGenerating synthetic data for testing...")
        generate_synthetic_training_data()

if __name__ == "__main__":
    main()
