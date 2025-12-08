"""
Synthetic Document Dataset Generator
=====================================
Generates synthetic training data for document fraud detection.
Creates augmented versions of sample documents for training.

Usage:
    python generate_dataset.py --output_dir ./document_dataset --samples 100
"""

import os
import sys
import argparse
from pathlib import Path
import random
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# ============================================================
# DOCUMENT TEMPLATES
# ============================================================

def create_id_card_template(width=600, height=400, is_fake=False):
    """Create a synthetic ID card image."""
    
    # Background color
    if is_fake:
        bg_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
    else:
        bg_color = (240, 240, 250)
    
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Add border
    border_color = (100, 100, 150) if not is_fake else (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
    draw.rectangle([5, 5, width-5, height-5], outline=border_color, width=3)
    
    # Add header
    header_text = "IDENTIFICATION CARD" if not is_fake else random.choice(["SAMPLE", "SPECIMEN", "FAKE ID", "TEST DOCUMENT"])
    try:
        font = ImageFont.truetype("arial.ttf", 24)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    draw.text((width//2 - 100, 20), header_text, fill=(50, 50, 100), font=font)
    
    # Add photo placeholder
    photo_x, photo_y = 30, 70
    photo_w, photo_h = 120, 150
    draw.rectangle([photo_x, photo_y, photo_x + photo_w, photo_y + photo_h], 
                   fill=(200, 200, 200), outline=(100, 100, 100))
    draw.text((photo_x + 30, photo_y + 60), "PHOTO", fill=(150, 150, 150), font=small_font)
    
    # Add text fields
    if is_fake:
        # Use suspicious placeholder data
        name = random.choice(["JOHN DOE", "JANE DOE", "SAMPLE PERSON", "TEST USER"])
        dob = random.choice(["01/01/1990", "00/00/0000", "XX/XX/XXXX"])
        id_num = random.choice(["123456789", "000000000", "111111111", "SAMPLE123"])
    else:
        # Generate more realistic data
        first_names = ["MICHAEL", "SARAH", "DAVID", "EMMA", "JAMES", "OLIVIA"]
        last_names = ["JOHNSON", "WILLIAMS", "BROWN", "JONES", "MILLER", "DAVIS"]
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        dob = f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.randint(1960, 2000)}"
        id_num = ''.join(random.choices(string.digits, k=9))
    
    text_x = 170
    draw.text((text_x, 80), f"Name: {name}", fill=(30, 30, 30), font=small_font)
    draw.text((text_x, 110), f"DOB: {dob}", fill=(30, 30, 30), font=small_font)
    draw.text((text_x, 140), f"ID: {id_num}", fill=(30, 30, 30), font=small_font)
    
    # Add fake indicators for fake documents
    if is_fake:
        # Add visible watermark
        watermark_text = random.choice(["SAMPLE", "VOID", "NOT VALID", "SPECIMEN"])
        draw.text((width//2 - 80, height//2), watermark_text, 
                  fill=(255, 200, 200), font=font)
        
        # Add imperfections
        if random.random() > 0.5:
            # Add noise
            img_array = np.array(img)
            noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
    
    return img


def create_license_template(width=600, height=380, is_fake=False):
    """Create a synthetic driver's license image."""
    
    bg_color = (230, 240, 250) if not is_fake else (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 12)
        title_font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()
        small_font = font
        title_font = font
    
    # State header
    state = random.choice(["CALIFORNIA", "NEW YORK", "TEXAS", "FLORIDA"])
    draw.text((20, 15), state, fill=(0, 50, 100), font=title_font)
    draw.text((20, 50), "DRIVER LICENSE", fill=(0, 50, 100), font=font)
    
    # Photo area
    photo_x, photo_y = 20, 90
    draw.rectangle([photo_x, photo_y, photo_x + 100, photo_y + 130], 
                   fill=(180, 180, 180), outline=(100, 100, 100))
    
    # License details
    text_x = 140
    
    if is_fake:
        lic_num = random.choice(["DL123456789", "SAMPLE12345", "000000000"])
        name = random.choice(["DOE, JOHN", "DOE, JANE", "SAMPLE, TEST"])
        addr = random.choice(["123 MAIN ST", "123 FAKE ST", "ANYTOWN, USA"])
        exp_date = random.choice(["01/01/2030", "XX/XX/XXXX", "00/00/0000"])
    else:
        lic_num = 'DL' + ''.join(random.choices(string.digits, k=8))
        first = random.choice(["SMITH", "JOHNSON", "WILLIAMS", "BROWN"])
        last = random.choice(["MICHAEL", "SARAH", "DAVID", "EMMA"])
        name = f"{first}, {last}"
        addr = f"{random.randint(100, 9999)} {random.choice(['OAK', 'MAPLE', 'ELM', 'PINE'])} {random.choice(['ST', 'AVE', 'DR', 'RD'])}"
        exp_date = f"{random.randint(1, 12):02d}/{random.randint(1, 28):02d}/{random.randint(2025, 2030)}"
    
    draw.text((text_x, 90), f"DL: {lic_num}", fill=(0, 0, 0), font=small_font)
    draw.text((text_x, 115), f"NAME: {name}", fill=(0, 0, 0), font=small_font)
    draw.text((text_x, 140), f"ADDR: {addr}", fill=(0, 0, 0), font=small_font)
    draw.text((text_x, 165), f"EXP: {exp_date}", fill=(0, 0, 0), font=small_font)
    draw.text((text_x, 190), f"CLASS: {random.choice(['C', 'D', 'A', 'B'])}", fill=(0, 0, 0), font=small_font)
    
    # Fake indicators
    if is_fake:
        watermark = random.choice(["SAMPLE", "VOID", "SPECIMEN", "NOT VALID"])
        # Semi-transparent watermark
        draw.text((width//2 - 60, height//2 - 20), watermark, fill=(255, 100, 100), font=title_font)
    
    return img


def apply_augmentation(img, is_fake=False):
    """Apply random augmentations to an image."""
    
    # Random rotation
    if random.random() > 0.5:
        angle = random.uniform(-10, 10)
        img = img.rotate(angle, fillcolor=(255, 255, 255))
    
    # Random brightness
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Random contrast
    if random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Add blur for fake documents (simulating low quality)
    if is_fake and random.random() > 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2)))
    
    # Add JPEG artifacts
    if random.random() > 0.5:
        from io import BytesIO
        buffer = BytesIO()
        quality = random.randint(60, 95) if not is_fake else random.randint(30, 70)
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img = Image.open(buffer)
    
    return img


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(output_dir, num_samples=100):
    """Generate synthetic document dataset."""
    
    output_path = Path(output_dir)
    real_dir = output_path / 'real'
    fake_dir = output_path / 'fake'
    
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("GENERATING SYNTHETIC DOCUMENT DATASET")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Samples per class: {num_samples}")
    
    # Generate real documents
    print(f"\n[1/2] Generating {num_samples} REAL documents...")
    for i in range(num_samples):
        if random.random() > 0.5:
            img = create_id_card_template(is_fake=False)
        else:
            img = create_license_template(is_fake=False)
        
        img = apply_augmentation(img, is_fake=False)
        
        filename = f"real_doc_{i:04d}.jpg"
        img.save(real_dir / filename, 'JPEG', quality=90)
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_samples}")
    
    # Generate fake documents
    print(f"\n[2/2] Generating {num_samples} FAKE documents...")
    for i in range(num_samples):
        if random.random() > 0.5:
            img = create_id_card_template(is_fake=True)
        else:
            img = create_license_template(is_fake=True)
        
        img = apply_augmentation(img, is_fake=True)
        
        filename = f"fake_doc_{i:04d}.jpg"
        img.save(fake_dir / filename, 'JPEG', quality=75)
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_samples}")
    
    print(f"\n{'='*60}")
    print("DATASET GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Real documents: {num_samples} ({real_dir})")
    print(f"Fake documents: {num_samples} ({fake_dir})")
    print(f"Total: {num_samples * 2} images")
    print(f"\nTo train the model, run:")
    print(f"  python train_document_model.py --data_dir {output_dir}")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Synthetic Document Dataset')
    parser.add_argument('--output_dir', type=str, default='./document_dataset',
                        help='Output directory for dataset')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples per class (real/fake)')
    
    args = parser.parse_args()
    
    generate_dataset(
        output_dir=args.output_dir,
        num_samples=args.samples
    )
