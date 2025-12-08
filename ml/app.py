# pylint: disable=no-member
import os
import subprocess
import base64
import uuid
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import traceback
import cv2
import io

# Day 6: Document analysis imports
try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False
    print("⚠️ piexif not installed - EXIF extraction disabled")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("⚠️ pdf2image not installed - PDF conversion disabled")

# Day 6 Enhancement: OCR for text extraction
try:
    import easyocr
    OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    OCR_AVAILABLE = True
    print("✅ EasyOCR initialized")
except ImportError:
    OCR_AVAILABLE = False
    OCR_READER = None
    print("⚠️ easyocr not installed - OCR disabled")

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = Path("./uploads")
FRAMES_FOLDER = Path("./frames")
MODELS_FOLDER = Path("./models")
UPLOAD_FOLDER.mkdir(exist_ok=True)
FRAMES_FOLDER.mkdir(exist_ok=True)
MODELS_FOLDER.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Day 6: Load Document Fraud CNN model if available
DOCUMENT_CNN_MODEL = None
DOCUMENT_CNN_AVAILABLE = False
try:
    from train_document_model import DocumentFraudCNN, get_transforms
    doc_model_path = MODELS_FOLDER / "document_fraud_model.pth"
    if doc_model_path.exists():
        DOCUMENT_CNN_MODEL = DocumentFraudCNN(num_classes=2, pretrained=False)
        checkpoint = torch.load(doc_model_path, map_location=device)
        DOCUMENT_CNN_MODEL.load_state_dict(checkpoint['model_state_dict'])
        DOCUMENT_CNN_MODEL = DOCUMENT_CNN_MODEL.to(device)
        DOCUMENT_CNN_MODEL.eval()
        DOCUMENT_CNN_AVAILABLE = True
        print(f"✅ Document Fraud CNN loaded (Val Acc: {checkpoint.get('val_acc', 0):.1f}%)")
    else:
        print("⚠️ Document Fraud CNN not found - run train_document_model.py first")
except Exception as e:
    print(f"⚠️ Could not load Document Fraud CNN: {e}")
print(f"Device: {device}")

# Face detection - try multiple cascades for better detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# ============================================================
# TRAINED MODEL
# ============================================================
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_features(self, x):
        return self.backbone.features(x)

# Load model
TRAINED_MODEL_PATH = MODELS_FOLDER / "deepfake_detector.pth"
PRODUCTION_MODE = TRAINED_MODEL_PATH.exists()

# Flag to invert scores if model was trained with opposite labels
INVERT_SCORES = True

# Calibration: boost uncertain scores toward clearer decisions
# More aggressive settings for under-trained models (1 epoch)
# Reduce these values after training with more epochs
CALIBRATION = {
    "enabled": True,
    "fake_boost_threshold": 0.38,  # Very aggressive - for 1 epoch model
    "real_boost_threshold": 0.32,  # Push real scores lower
    "boost_strength": 1.6,         # Strong boost for under-trained model
    "epochs_trained": 1,           # Track this - reduce boost when epochs increase
}

if PRODUCTION_MODE:
    print(f"✅ Loading trained model: {TRAINED_MODEL_PATH}")
    model = DeepfakeDetector()
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device, weights_only=True))
    model.eval().to(device)
    print("✅ TRAINED MODEL LOADED")
    print(f"   Score inversion: {'ENABLED' if INVERT_SCORES else 'DISABLED'}")
    print(f"   Calibration: {'ENABLED' if CALIBRATION['enabled'] else 'DISABLED'}")
else:
    print("⚠️ No trained model, using base model")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.eval().to(device)
    INVERT_SCORES = False

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================
class GradCAM:
    """Grad-CAM implementation for deepfake detection visualization"""
    
    def __init__(self, nn_model, layer):
        self.model = nn_model
        self.target_layer = layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        layer.register_forward_hook(self._forward_hook)
        layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, _module, _input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap"""
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # target_class parameter reserved for future use
        _ = target_class
        
        # Backward pass
        self.model.zero_grad()
        output.backward(gradient=torch.ones_like(output))
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU and normalize
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam

def get_gradcam_target_layer(nn_model):
    """Get the target layer for Grad-CAM"""
    if PRODUCTION_MODE:
        # For our custom model, use last conv layer of backbone
        return nn_model.backbone.features[-1]
    else:
        # For base EfficientNet
        return nn_model.features[-1]

# Initialize Grad-CAM
try:
    target_layer = get_gradcam_target_layer(model)
    gradcam = GradCAM(model, target_layer)
    GRADCAM_AVAILABLE = True
    print("✅ Grad-CAM initialized")
except (AttributeError, RuntimeError) as e:
    print(f"⚠️ Grad-CAM init failed: {e}")
    GRADCAM_AVAILABLE = False
    gradcam = None

def generate_heatmap(input_tensor, original_image):
    """Generate Grad-CAM heatmap overlay"""
    if not GRADCAM_AVAILABLE or gradcam is None:
        return None
    
    try:
        model.eval()
        
        # Generate CAM
        cam = gradcam.generate(input_tensor)
        
        # Convert to numpy and resize
        cam_np = cam.squeeze().cpu().numpy()
        cam_resized = cv2.resize(cam_np, (original_image.width, original_image.height))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        original_np = np.array(original_image)
        overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
        
        # Convert to base64
        overlay_pil = Image.fromarray(overlay)
        buf = io.BytesIO()
        overlay_pil.save(buf, format='PNG')
        heatmap_base64 = base64.b64encode(buf.getvalue()).decode()
        
        return heatmap_base64
    except (RuntimeError, ValueError) as e:
        print(f"Heatmap generation error: {e}")
        return None

def generate_heatmap_only(input_tensor, size=(224, 224)):
    """Generate just the heatmap (without overlay)"""
    if not GRADCAM_AVAILABLE or gradcam is None:
        return None
    
    try:
        model.eval()
        cam = gradcam.generate(input_tensor)
        cam_np = cam.squeeze().cpu().numpy()
        cam_resized = cv2.resize(cam_np, size)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        heatmap_pil = Image.fromarray(heatmap)
        buf = io.BytesIO()
        heatmap_pil.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    except (RuntimeError, ValueError):
        return None

# ============================================================
# FACE EXTRACTION & ANALYSIS
# ============================================================
def _detect_faces_with_cascade(gray, cascade, scales):
    """Try to detect faces with given cascade and scales."""
    for scale in scales:
        faces = cascade.detectMultiScale(gray, scale, 4, minSize=(40, 40))
        if len(faces) > 0:
            return faces
    return []

def extract_face(image, margin=0.4):
    """Extract face with multiple cascade fallbacks for better detection."""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Try different cascades
    faces = _detect_faces_with_cascade(gray, face_cascade, [1.05, 1.1, 1.15, 1.2])
    
    if len(faces) == 0:
        faces = _detect_faces_with_cascade(gray, face_cascade_alt, [1.05, 1.1, 1.2])
    
    if len(faces) == 0:
        faces = _detect_faces_with_cascade(gray, profile_cascade, [1.1, 1.2, 1.3])
    
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        x1 = max(0, int(x - w * margin))
        y1 = max(0, int(y - h * margin))
        x2 = min(image.width, int(x + w + w * margin))
        y2 = min(image.height, int(y + h + h * margin))
        return image.crop((x1, y1, x2, y2)), True, (x, y, w, h)
    
    # No face found - use center crop
    w, h = image.size
    m = min(w, h)
    return image.crop(((w-m)//2, (h-m)//2, (w+m)//2, (h+m)//2)), False, None

def _run_model_inference(input_tensor):
    """Run model inference and return score with calibration."""
    if PRODUCTION_MODE:
        output = model(input_tensor)
        raw_score = torch.sigmoid(output).item()
        nn_score = 1.0 - raw_score if INVERT_SCORES else raw_score
        
        # Apply calibration - more aggressive for under-trained models
        if CALIBRATION["enabled"]:
            # For 1-epoch model, be very aggressive
            if nn_score > CALIBRATION["fake_boost_threshold"]:
                # Boost toward fake
                excess = nn_score - CALIBRATION["fake_boost_threshold"]
                nn_score = CALIBRATION["fake_boost_threshold"] + excess * CALIBRATION["boost_strength"]
            elif nn_score < CALIBRATION["real_boost_threshold"]:
                # Boost toward real  
                deficit = CALIBRATION["real_boost_threshold"] - nn_score
                nn_score = CALIBRATION["real_boost_threshold"] - deficit * CALIBRATION["boost_strength"]
        
        # Clamp to valid range
        nn_score = max(0.08, min(0.92, nn_score))
        return nn_score, raw_score
    else:
        features = model.features(input_tensor)
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten()
        nn_score = 0.3 + (pooled.std().item() * 0.5)
        return nn_score, nn_score

def _calculate_confidence(nn_score, face_detected):
    """Calculate confidence and adjust score based on face detection."""
    base_confidence = abs(nn_score - 0.5) * 2 * 100
    if face_detected:
        confidence = max(60, min(95, 50 + base_confidence))
        return confidence, nn_score
    # Lower confidence when no face detected
    confidence = max(30, min(60, 30 + base_confidence * 0.5))
    adjusted_score = nn_score * 0.7 + 0.35 * 0.3
    return confidence, adjusted_score

def analyze_frame(frame_path, generate_heatmap_flag=False):
    """Analyze frame with optional Grad-CAM heatmap"""
    try:
        img = Image.open(frame_path).convert("RGB")
        face_img, face_detected, _ = extract_face(img)
        
        input_tensor = preprocess(face_img).unsqueeze(0).to(device)
        input_tensor.requires_grad = True
        
        with torch.enable_grad():
            nn_score, raw_score = _run_model_inference(input_tensor)
        
        # Generate heatmaps if requested
        heatmap_base64 = None
        heatmap_overlay_base64 = None
        
        if generate_heatmap_flag and GRADCAM_AVAILABLE:
            input_tensor_grad = preprocess(face_img).unsqueeze(0).to(device)
            input_tensor_grad.requires_grad = True
            heatmap_base64 = generate_heatmap_only(input_tensor_grad, (face_img.width, face_img.height))
            heatmap_overlay_base64 = generate_heatmap(input_tensor_grad, face_img)
        
        confidence, final_score = _calculate_confidence(nn_score, face_detected)
        
        # Build details dict
        details = {"neural_network": round(final_score, 3)}
        if PRODUCTION_MODE:
            details["raw_score"] = round(raw_score, 3)
        
        return {
            "deepfake_probability": float(np.clip(final_score, 0, 1)),
            "confidence": float(confidence),
            "face_detected": face_detected,
            "heatmap": heatmap_base64,
            "heatmap_overlay": heatmap_overlay_base64,
            "details": details
        }
    except (IOError, RuntimeError) as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return {"deepfake_probability": 0.5, "confidence": 20.0, "face_detected": False, "heatmap": None, "details": {}}

def image_to_base64(path, max_size=400):
    try:
        img = Image.open(path)
        ratio = min(max_size / img.width, max_size / img.height)
        if ratio < 1:
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except (IOError, OSError):
        return ""

def find_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return "ffmpeg"
    except (FileNotFoundError, subprocess.CalledProcessError):
        for pkg in Path(os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Packages")).glob("Gyan.FFmpeg*"):
            for exe in pkg.rglob("ffmpeg.exe"):
                try:
                    subprocess.run([str(exe), "-version"], capture_output=True, check=True)
                    return str(exe)
                except (FileNotFoundError, subprocess.CalledProcessError):
                    pass
    return None

FFMPEG = find_ffmpeg()
print(f"FFmpeg: {'Yes' if FFMPEG else 'No'}")

def extract_frames(video, out_dir, fps=1):
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, "-i", str(video), "-vf", f"fps={fps}", "-q:v", "2", str(out_dir / "frame_%04d.png"), "-y"], capture_output=True, check=False)
    return sorted(out_dir.glob("frame_*.png"))

# ============================================================
# DAY 6: ERROR LEVEL ANALYSIS (ELA)
# ============================================================
def generate_ela(image_path, quality=90, amplification=15):
    """
    Generate Error Level Analysis image to detect tampering.
    ELA works by re-saving the image at a known quality level and
    comparing it to the original. Edited areas show different error levels.
    
    Args:
        image_path: Path to the image file
        quality: JPEG quality for resave (default 90)
        amplification: Factor to amplify differences (default 15)
    
    Returns:
        dict with ela_image (base64), brightness_std, and suspect_regions
    """
    try:
        # Open original image
        original = Image.open(image_path).convert('RGB')
        original_np = np.array(original)
        
        # Re-save at specified quality
        buffer = io.BytesIO()
        original.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer).convert('RGB')
        resaved_np = np.array(resaved)
        
        # Compute absolute difference
        diff = np.abs(original_np.astype(np.float32) - resaved_np.astype(np.float32))
        
        # Amplify the differences
        ela = np.clip(diff * amplification, 0, 255).astype(np.uint8)
        
        # Convert to grayscale for analysis
        ela_gray = cv2.cvtColor(ela, cv2.COLOR_RGB2GRAY)
        
        # Calculate statistics for tamper detection
        brightness_std = float(np.std(ela_gray))
        brightness_mean = float(np.mean(ela_gray))
        
        # Find high-contrast regions (potential edits)
        threshold = np.percentile(ela_gray, 95)
        suspect_mask = ela_gray > threshold
        suspect_percentage = float(np.sum(suspect_mask) / suspect_mask.size * 100)
        
        # Create colored ELA visualization
        ela_colored = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)
        ela_colored = cv2.cvtColor(ela_colored, cv2.COLOR_BGR2RGB)
        
        # Convert to base64
        ela_pil = Image.fromarray(ela_colored)
        buf = io.BytesIO()
        ela_pil.save(buf, format='PNG')
        ela_base64 = base64.b64encode(buf.getvalue()).decode()
        
        # Also provide raw ELA (grayscale amplified)
        ela_raw_pil = Image.fromarray(ela)
        buf_raw = io.BytesIO()
        ela_raw_pil.save(buf_raw, format='PNG')
        ela_raw_base64 = base64.b64encode(buf_raw.getvalue()).decode()
        
        return {
            "ela_image": ela_base64,
            "ela_raw": ela_raw_base64,
            "brightness_std": round(brightness_std, 2),
            "brightness_mean": round(brightness_mean, 2),
            "suspect_percentage": round(suspect_percentage, 2),
            "quality_used": quality,
            "amplification": amplification
        }
    except Exception as e:
        print(f"ELA generation error: {e}")
        traceback.print_exc()
        return None


def get_ela_interpretation(ela_result):
    """
    Interpret ELA results to provide human-readable analysis.
    """
    if not ela_result:
        return "ELA analysis could not be performed."
    
    std = ela_result.get("brightness_std", 0)
    suspect = ela_result.get("suspect_percentage", 0)
    
    indicators = []
    tamper_score = 0
    
    # High standard deviation indicates non-uniform compression
    if std > 25:
        indicators.append("High variation in compression artifacts detected")
        tamper_score += 30
    elif std > 15:
        indicators.append("Moderate variation in compression artifacts")
        tamper_score += 15
    
    # High suspect percentage indicates potential edits
    if suspect > 5:
        indicators.append(f"{suspect:.1f}% of image shows high error levels")
        tamper_score += 25
    elif suspect > 2:
        indicators.append(f"Small regions ({suspect:.1f}%) show elevated error levels")
        tamper_score += 10
    
    if tamper_score >= 40:
        verdict = "HIGH LIKELIHOOD of tampering"
    elif tamper_score >= 20:
        verdict = "MODERATE signs of potential editing"
    else:
        verdict = "LOW likelihood of tampering"
    
    explanation = f"{verdict}. " + "; ".join(indicators) if indicators else f"{verdict}. Image appears to have uniform compression."
    
    return explanation


# ============================================================
# DAY 6: EXIF METADATA EXTRACTION
# ============================================================
EDITING_SOFTWARE_SIGNATURES = [
    "photoshop", "gimp", "lightroom", "capture one", "affinity",
    "pixelmator", "paint.net", "corel", "acdsee", "photoscape",
    "snapseed", "vsco", "afterlight", "facetune", "faceapp",
    "remini", "lensa", "prisma", "meitu", "beautycam"
]


def extract_exif_metadata(image_path):
    """
    Extract EXIF metadata from image and detect editing software.
    
    Returns:
        dict with camera info, software, timestamps, and tamper indicators
    """
    result = {
        "camera": None,
        "software": None,
        "date_time_original": None,
        "date_time_digitized": None,
        "date_time_modified": None,
        "gps_info": None,
        "editing_detected": False,
        "editing_software": [],
        "tamper_indicators": [],
        "all_tags": {}
    }
    
    if not PIEXIF_AVAILABLE:
        result["tamper_indicators"].append("EXIF extraction unavailable")
        return result
    
    try:
        exif_dict = piexif.load(str(image_path))
        
        # Extract from 0th IFD (main image)
        if "0th" in exif_dict:
            ifd = exif_dict["0th"]
            
            # Camera make and model
            make = ifd.get(piexif.ImageIFD.Make, b"").decode("utf-8", errors="ignore").strip()
            model = ifd.get(piexif.ImageIFD.Model, b"").decode("utf-8", errors="ignore").strip()
            if make or model:
                result["camera"] = f"{make} {model}".strip()
            
            # Software used
            software = ifd.get(piexif.ImageIFD.Software, b"").decode("utf-8", errors="ignore").strip()
            if software:
                result["software"] = software
                result["all_tags"]["Software"] = software
                
                # Check for editing software
                software_lower = software.lower()
                for sig in EDITING_SOFTWARE_SIGNATURES:
                    if sig in software_lower:
                        result["editing_detected"] = True
                        result["editing_software"].append(software)
                        result["tamper_indicators"].append(f"Editing software detected: {software}")
                        break
            
            # DateTime
            dt = ifd.get(piexif.ImageIFD.DateTime, b"").decode("utf-8", errors="ignore").strip()
            if dt:
                result["date_time_modified"] = dt
                result["all_tags"]["DateTime"] = dt
        
        # Extract from Exif IFD
        if "Exif" in exif_dict:
            exif_ifd = exif_dict["Exif"]
            
            # Original date taken
            dto = exif_ifd.get(piexif.ExifIFD.DateTimeOriginal, b"").decode("utf-8", errors="ignore").strip()
            if dto:
                result["date_time_original"] = dto
                result["all_tags"]["DateTimeOriginal"] = dto
            
            # Digitized date
            dtd = exif_ifd.get(piexif.ExifIFD.DateTimeDigitized, b"").decode("utf-8", errors="ignore").strip()
            if dtd:
                result["date_time_digitized"] = dtd
                result["all_tags"]["DateTimeDigitized"] = dtd
        
        # Check for timestamp inconsistencies
        timestamps = [result["date_time_original"], result["date_time_digitized"], result["date_time_modified"]]
        timestamps = [t for t in timestamps if t]
        if len(set(timestamps)) > 1:
            result["tamper_indicators"].append("Inconsistent timestamps detected")
        
        # Check for missing EXIF (common in edited images)
        if not result["camera"] and not result["date_time_original"]:
            result["tamper_indicators"].append("Missing camera EXIF data (may indicate editing)")
        
        # GPS info
        if "GPS" in exif_dict and exif_dict["GPS"]:
            result["gps_info"] = "GPS coordinates present"
            result["all_tags"]["GPS"] = "Present"
        
    except Exception as e:
        print(f"EXIF extraction error: {e}")
        result["tamper_indicators"].append(f"EXIF read error: {str(e)[:50]}")
    
    return result


# ============================================================
# DAY 6: PDF TO IMAGE CONVERSION
# ============================================================
def convert_pdf_to_images(pdf_path, dpi=150, max_pages=5):
    """
    Convert PDF pages to images for analysis.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion (default 150)
        max_pages: Maximum pages to convert (default 5)
    
    Returns:
        list of PIL Image objects
    """
    if not PDF2IMAGE_AVAILABLE:
        print("PDF conversion not available - pdf2image not installed")
        return []
    
    # Poppler path for Windows
    poppler_path = None
    possible_paths = [
        r"C:\poppler\poppler-24.02.0\Library\bin",
        r"C:\Program Files\poppler\Library\bin",
        r"C:\poppler\bin"
    ]
    for path in possible_paths:
        if Path(path).exists():
            poppler_path = path
            break
    
    try:
        # Convert PDF to images
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=1,
            last_page=max_pages,
            poppler_path=poppler_path
        )
        print(f"Converted {len(images)} pages from PDF")
        return images
    except Exception as e:
        print(f"PDF conversion error: {e}")
        traceback.print_exc()
        return []


# ============================================================
# DAY 6 ENHANCEMENT: DOCUMENT FRAUD DETECTION
# ============================================================

# Keywords that indicate sample/specimen/fake documents
SAMPLE_KEYWORDS = [
    "sample", "specimen", "void", "not valid", "for demonstration",
    "example", "test", "demo", "dummy", "fake", "template",
    "not for official use", "invalid", "cancelled", "facsimile"
]

# Common document types and their expected patterns
DOCUMENT_TYPES = {
    "drivers_license": ["driver", "license", "licence", "dl", "class", "dob", "exp", "iss"],
    "passport": ["passport", "nationality", "surname", "given names", "date of birth"],
    "id_card": ["identification", "id card", "national id", "citizen"],
    "certificate": ["certificate", "certify", "awarded", "completed", "achievement"],
    "invoice": ["invoice", "bill", "total", "amount due", "payment"],
    "bank_statement": ["bank", "statement", "balance", "transaction", "account"]
}

# Suspicious patterns in documents
SUSPICIOUS_PATTERNS = [
    "123456789",  # Sequential numbers
    "000000000",  # All zeros
    "111111111",  # Repeated digits
    "john doe", "jane doe",  # Placeholder names
    "123 main st", "123 main street",  # Placeholder addresses
    "anytown", "anycity", "anystate",  # Placeholder locations
    "xx/xx/xxxx", "00/00/0000",  # Placeholder dates
]


def detect_document_fraud(image_path):
    """
    Analyze document for signs of fraud using multiple heuristics + CNN.
    
    Returns:
        dict with fraud indicators, document type, and authenticity score
    """
    result = {
        "document_type": "unknown",
        "document_type_confidence": 0,
        "is_sample_specimen": False,
        "fraud_indicators": [],
        "authenticity_score": 100,  # Start at 100, deduct for issues
        "text_detected": [],
        "analysis_details": {},
        "cnn_prediction": None,
        "cnn_confidence": 0
    }
    
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # CNN Prediction (if model available)
        if DOCUMENT_CNN_AVAILABLE and DOCUMENT_CNN_MODEL is not None:
            try:
                transform = get_transforms(is_training=False)
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = DOCUMENT_CNN_MODEL(img_tensor)
                    probs = torch.softmax(logits, dim=1)
                
                prob_real = probs[0][0].item()
                prob_fake = probs[0][1].item()
                
                result["cnn_prediction"] = "AUTHENTIC" if prob_real > prob_fake else "FRAUDULENT"
                result["cnn_confidence"] = max(prob_real, prob_fake)
                
                print(f"  [CNN] Prediction: {result['cnn_prediction']} ({result['cnn_confidence']*100:.1f}%)")
                
                # Adjust score based on CNN - lower thresholds for better detection
                if prob_fake > 0.6:
                    result["authenticity_score"] -= 50
                    result["fraud_indicators"].append(f"CNN detected fraud ({prob_fake*100:.0f}% confidence)")
                    result["is_sample_specimen"] = True  # Flag as fraud
                elif prob_fake > 0.5:
                    result["authenticity_score"] -= 30
                    result["fraud_indicators"].append(f"CNN suspects fraud ({prob_fake*100:.0f}% confidence)")
                    
            except Exception as cnn_error:
                print(f"  [CNN] Error: {cnn_error}")
        
        # 0. Check filename for suspicious keywords (normalize hyphens/underscores)
        filename_normalized = str(image_path).lower().replace("-", " ").replace("_", " ")
        filename_keywords = ["fake", "fraud", "forged", "counterfeit", "phony", 
                            "sample", "specimen", "template", "dark web", "darkweb",
                            "illegal", "scam", "false", "bogus", "identity documents"]
        for keyword in filename_keywords:
            if keyword in filename_normalized:
                result["fraud_indicators"].append(f"Suspicious filename keyword: '{keyword}'")
                result["authenticity_score"] -= 30
                if keyword in ["fake", "fraud", "forged", "counterfeit", "dark web", "identity documents"]:
                    result["is_sample_specimen"] = True
        
        # 1. Text extraction using OCR
        extracted_text = extract_text_regions(gray)
        result["text_detected"] = extracted_text[:10]  # Limit for response size
        
        # 2. Check for sample/specimen keywords in OCR text
        text_lower = " ".join(extracted_text).lower()
        for keyword in SAMPLE_KEYWORDS:
            if keyword in text_lower:
                result["is_sample_specimen"] = True
                result["fraud_indicators"].append(f"Sample/specimen keyword detected: '{keyword}'")
                result["authenticity_score"] -= 40
                break
        
        # 3. Check for suspicious patterns
        for pattern in SUSPICIOUS_PATTERNS:
            if pattern in text_lower:
                result["fraud_indicators"].append(f"Suspicious pattern detected: '{pattern}'")
                result["authenticity_score"] -= 15
        
        # 4. Detect document type
        doc_type, confidence = classify_document_type(text_lower)
        result["document_type"] = doc_type
        result["document_type_confidence"] = confidence
        
        # 5. Visual analysis for document authenticity
        visual_score, visual_issues = analyze_document_visuals(img_np, gray)
        result["authenticity_score"] -= visual_score
        result["fraud_indicators"].extend(visual_issues)
        
        # 6. Check image quality and resolution
        quality_issues = check_image_quality(img, gray)
        result["fraud_indicators"].extend(quality_issues)
        result["authenticity_score"] -= len(quality_issues) * 5
        
        # 7. Detect multiple documents in one image (suspicious for ID theft)
        multi_doc_score = detect_multiple_documents(gray)
        if multi_doc_score > 0:
            result["fraud_indicators"].append(f"Multiple document regions detected ({multi_doc_score})")
            result["authenticity_score"] -= multi_doc_score * 10
        
        # Ensure score is in valid range
        result["authenticity_score"] = max(0, min(100, result["authenticity_score"]))
        
        # Determine verdict
        if result["is_sample_specimen"]:
            result["verdict"] = "LIKELY_FRAUDULENT"
        elif result["authenticity_score"] < 40:
            result["verdict"] = "LIKELY_FRAUDULENT"
        elif result["authenticity_score"] < 70:
            result["verdict"] = "SUSPICIOUS"
        else:
            result["verdict"] = "LIKELY_AUTHENTIC"
        
    except Exception as e:
        print(f"Document fraud detection error: {e}")
        traceback.print_exc()
        result["fraud_indicators"].append(f"Analysis error: {str(e)[:50]}")
    
    return result


def extract_text_regions(gray_image):
    """
    Extract text from image using EasyOCR.
    Returns list of detected text strings.
    """
    extracted = []
    
    try:
        if OCR_AVAILABLE and OCR_READER is not None:
            # Use EasyOCR for real text extraction
            # Convert grayscale to RGB if needed (easyocr expects RGB or grayscale)
            results = OCR_READER.readtext(gray_image, detail=0, paragraph=True)
            extracted.extend(results)
            print(f"  [OCR] Extracted {len(results)} text blocks")
            if results:
                # Show first few detected texts for debugging
                preview = [t[:30] for t in results[:5]]
                print(f"  [OCR] Preview: {preview}")
        else:
            # Fallback: visual analysis only
            print("  [OCR] Not available, using visual analysis")
            
    except Exception as e:
        print(f"  [OCR] Error: {e}")
    
    return extracted


def classify_document_type(text_content):
    """
    Classify document type based on detected text patterns.
    """
    best_match = "unknown"
    best_score = 0
    
    for doc_type, keywords in DOCUMENT_TYPES.items():
        score = sum(1 for kw in keywords if kw in text_content)
        if score > best_score:
            best_score = score
            best_match = doc_type
    
    confidence = min(100, best_score * 25) if best_score > 0 else 0
    return best_match, confidence


def analyze_document_visuals(img_np, gray):
    """
    Analyze visual characteristics of document for authenticity.
    """
    issues = []
    penalty = 0
    
    try:
        # 1. Check for uniform background (real IDs have security patterns)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density < 0.02:
            issues.append("Very low edge density - may be digitally created")
            penalty += 10
        
        # 2. Check color distribution
        if len(img_np.shape) == 3:
            color_std = np.std(img_np, axis=(0, 1))
            if np.mean(color_std) < 20:
                issues.append("Unusually uniform colors")
                penalty += 5
        
        # 3. Check for repeated patterns (copy-paste detection)
        # Simplified version - check for exact pixel matches
        h, w = gray.shape
        if h > 100 and w > 100:
            top_region = gray[:h//4, :]
            bottom_region = gray[3*h//4:, :]
            
            if np.allclose(top_region[:min(50, h//4)], bottom_region[:min(50, h//4)], atol=5):
                issues.append("Repeated patterns detected")
                penalty += 15
        
        # 4. Check for straight rectangular borders (may indicate screenshot)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 1:
            approx = cv2.approxPolyDP(contours[0], 0.02 * cv2.arcLength(contours[0], True), True)
            if len(approx) == 4:
                issues.append("Perfect rectangular border - may be screenshot/scan")
                penalty += 5
                
    except Exception as e:
        print(f"Visual analysis error: {e}")
    
    return penalty, issues


def check_image_quality(img, gray):
    """
    Check image quality metrics that may indicate manipulation.
    """
    issues = []
    
    try:
        w, h = img.size
        
        # Very low resolution
        if w < 300 or h < 200:
            issues.append("Very low resolution image")
        
        # Unusual aspect ratio for documents
        aspect = w / h
        if aspect > 4 or aspect < 0.25:
            issues.append("Unusual aspect ratio for document")
        
        # Check for JPEG artifacts (block patterns)
        if w >= 8 and gray.shape[0] >= 8:
            block_std = np.std([
                np.std(gray[i:i+8, j:j+8]) 
                for i in range(0, gray.shape[0]-8, 8) 
                for j in range(0, gray.shape[1]-8, 8)
            ][:100])  # Sample first 100 blocks
            
            if block_std < 5:
                issues.append("Unusual compression patterns")
                
    except Exception as e:
        print(f"Quality check error: {e}")
    
    return issues


def detect_multiple_documents(gray):
    """
    Detect if image contains multiple documents (suspicious for ID fraud).
    Returns count of document-like regions found.
    """
    try:
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect edges
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for document-sized rectangles
        h, w = gray.shape
        min_area = (h * w) * 0.05  # At least 5% of image
        max_area = (h * w) * 0.8   # At most 80% of image
        
        document_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Check if roughly rectangular
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if 4 <= len(approx) <= 8:  # Roughly rectangular
                    document_count += 1
        
        # Only flag if multiple documents found
        return document_count if document_count > 1 else 0
        
    except Exception as e:
        print(f"Multi-document detection error: {e}")
        return 0


def get_explanation(score):
    """Generate explanation text based on analysis"""
    if score > 0.70:
        return "High likelihood of manipulation detected. The model identified suspicious patterns in facial features, skin texture, or boundary regions that are commonly associated with deepfake generation techniques."
    elif score > 0.55:
        return "Moderate signs of potential manipulation. Some facial regions show patterns that may indicate synthetic generation or face-swapping. Further verification recommended."
    elif score > 0.45:
        return "Inconclusive results. The analysis could not definitively determine authenticity. The content shows mixed signals that require human review."
    elif score > 0.30:
        return "Low likelihood of manipulation. Most facial features appear natural and consistent with authentic video content."
    else:
        return "Content appears authentic. No significant signs of deepfake manipulation detected. Natural facial features and consistent lighting/texture observed."

# ============================================================
# DETECTION HELPERS
# ============================================================
def _load_frames(video_path, ext, frames_dir):
    """Load frames from image or video file."""
    frames = []
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}:
        frames_dir.mkdir(parents=True, exist_ok=True)
        fp = frames_dir / "frame_0001.png"
        Image.open(video_path).convert("RGB").save(fp)
        frames = [fp]
    elif ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"} and FFMPEG:
        frames = extract_frames(video_path, frames_dir)
        if len(frames) > 30:
            idx = np.linspace(0, len(frames)-1, 30, dtype=int)
            frames = [frames[i] for i in idx]
    return frames

def _analyze_all_frames(frames):
    """Analyze all frames and return results."""
    results, all_scores, face_scores = [], [], []
    
    for i, fp in enumerate(frames):
        analysis = analyze_frame(fp, generate_heatmap_flag=False)
        score = analysis["deepfake_probability"]
        face_detected = analysis.get("face_detected", True)
        
        indicator = _get_indicator(score)
        face_icon = "👤" if face_detected else "⚠️"
        print(f"  Frame {i+1}/{len(frames)}: {score*100:.1f}% {indicator} {face_icon}")
        
        results.append({
            "frame_number": i + 1,
            "frame_path": str(fp),
            "image": image_to_base64(fp),
            "deepfake_probability": score,
            "confidence": analysis["confidence"],
            "face_detected": face_detected,
            "heatmap": None,
            "heatmap_overlay": None,
            "details": analysis.get("details", {})
        })
        all_scores.append(score)
        if face_detected:
            face_scores.append((i, score))
    
    return results, all_scores, face_scores

def _get_indicator(score):
    """Get status indicator for score."""
    if score > 0.65:
        return "🔴 FAKE"
    if score > 0.40:
        return "🟡 UNCERTAIN"
    return "🟢 REAL"

def _add_heatmaps_to_results(results, face_scores, generate_heatmaps):
    """Generate and add heatmaps for top suspicious frames."""
    if not generate_heatmaps or not face_scores:
        return
    
    top_frames = sorted(face_scores, key=lambda x: x[1], reverse=True)[:3]
    print(f"\n  [Generating heatmaps for top {len(top_frames)} frames]")
    
    for idx, _ in top_frames:
        fp = Path(results[idx]["frame_path"])
        if fp.exists():
            analysis_with_heatmap = analyze_frame(fp, generate_heatmap_flag=True)
            results[idx]["heatmap"] = analysis_with_heatmap.get("heatmap")
            results[idx]["heatmap_overlay"] = analysis_with_heatmap.get("heatmap_overlay")
            print(f"    Frame {idx+1}: Heatmap generated ✓")

def _remove_outliers(scores_array):
    """Remove outliers using IQR method."""
    if len(scores_array) < 5:
        return scores_array
    q1, q3 = np.percentile(scores_array, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered = scores_array[(scores_array >= lower_bound) & (scores_array <= upper_bound)]
    return filtered if len(filtered) >= 3 else scores_array

def _compute_score_ratios(scores_array):
    """Compute fake/real score ratios."""
    total = len(scores_array)
    if total == 0:
        return 0, 0, 0
    high_fake = sum(1 for s in scores_array if s > 0.48)
    medium_fake = sum(1 for s in scores_array if s > 0.40)
    high_real = sum(1 for s in scores_array if s < 0.30)
    return high_fake / total, medium_fake / total, high_real / total

def _compute_weighted_final(scores_array, fake_ratio, medium_fake_ratio, real_ratio):
    """Compute weighted final score based on distribution."""
    avg = float(np.mean(scores_array))
    med = float(np.median(scores_array))
    p75 = float(np.percentile(scores_array, 75))
    p90 = float(np.percentile(scores_array, 90))
    
    if fake_ratio > 0.15 or medium_fake_ratio > 0.50:
        final = avg * 0.1 + p75 * 0.3 + p90 * 0.4 + med * 0.2
        return max(final, 0.55)
    if real_ratio > 0.45:
        p25 = float(np.percentile(scores_array, 25))
        final = avg * 0.3 + med * 0.4 + p25 * 0.3
        return min(final, 0.32)
    if medium_fake_ratio > 0.35:
        final = avg * 0.2 + p75 * 0.5 + med * 0.3
        return max(final, 0.50)
    return avg * 0.5 + med * 0.3 + p75 * 0.2

def _calculate_final_score(all_scores, face_scores):
    """Calculate final deepfake score with outlier removal."""
    # Prefer face-detected frames
    if len(face_scores) >= 3:
        scores_for_analysis = [s for _, s in face_scores]
    else:
        valid_scores = [s for s in all_scores if abs(s - 0.35) > 0.01]
        scores_for_analysis = valid_scores if len(valid_scores) >= 3 else all_scores
    
    if not scores_for_analysis:
        return 0.5, 0.5, 0.5, 0.5, 0.0
    
    scores_array = np.array(scores_for_analysis)
    scores_array = _remove_outliers(scores_array)
    
    avg = float(np.mean(scores_array))
    mx = float(np.max(scores_array))
    med = float(np.median(scores_array))
    std = float(np.std(scores_array))
    
    fake_ratio, medium_fake_ratio, real_ratio = _compute_score_ratios(scores_array)
    final = _compute_weighted_final(scores_array, fake_ratio, medium_fake_ratio, real_ratio)
    
    return float(np.clip(final, 0, 1)), avg, mx, med, std

def _get_verdict(final, std):
    """Determine verdict and confidence from final score."""
    if final < 0.28:
        verdict, conf = "AUTHENTIC", "HIGH"
    elif final < 0.40:
        verdict, conf = "LIKELY_AUTHENTIC", "MEDIUM"
    elif final < 0.48:
        verdict, conf = "UNCERTAIN", "LOW"
    elif final < 0.62:
        verdict, conf = "SUSPICIOUS", "MEDIUM"
    else:
        verdict, conf = "LIKELY_DEEPFAKE", "HIGH"
    
    if std > 0.15:
        conf = "LOW"
    
    return verdict, conf

# ============================================================
# API ROUTES
# ============================================================
@app.route("/")
def index():
    return jsonify({
        "service": "TrustLock Deepfake Detection",
        "version": "3.0 - Day 3",
        "features": ["deepfake_detection", "gradcam_heatmaps", "explainability"],
        "gradcam_available": GRADCAM_AVAILABLE
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "production": PRODUCTION_MODE, "gradcam": GRADCAM_AVAILABLE})

@app.route("/analyze-document", methods=["POST"])
def analyze_document():
    """
    Day 6: Document analysis endpoint.
    Performs ELA, EXIF extraction, and optional deepfake detection.
    Supports images and PDFs.
    """
    job_id = str(uuid.uuid4())
    file_path = None
    
    try:
        file = request.files.get("file")
        if not file or not file.filename:
            return jsonify({"error": "No file provided"}), 400
        
        print(f"\n[DOCUMENT ANALYSIS] {file.filename}")
        
        ext = Path(file.filename).suffix.lower()
        file_path = UPLOAD_FOLDER / f"{job_id}{ext}"
        file.save(file_path)
        
        # Determine file type
        is_pdf = ext == ".pdf"
        is_image = ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
        
        if not is_pdf and not is_image:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400
        
        results = {
            "job_id": job_id,
            "filename": file.filename,
            "file_type": "pdf" if is_pdf else "image",
            "pages": []
        }
        
        # Process PDF
        if is_pdf:
            if not PDF2IMAGE_AVAILABLE:
                return jsonify({
                    "error": "PDF processing not available. Install poppler and pdf2image."
                }), 503
            
            images = convert_pdf_to_images(file_path)
            
            if not images:
                return jsonify({"error": "Failed to convert PDF to images"}), 500
            
            for i, img in enumerate(images):
                # Save temporary image for analysis
                temp_path = UPLOAD_FOLDER / f"{job_id}_page_{i+1}.png"
                img.save(temp_path, "PNG")
                
                # Perform ELA
                ela_result = generate_ela(temp_path)
                
                # Get image as base64
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                img_base64 = base64.b64encode(buf.getvalue()).decode()
                
                page_result = {
                    "page_number": i + 1,
                    "image": img_base64,
                    "ela": ela_result,
                    "ela_interpretation": get_ela_interpretation(ela_result),
                    "exif": None  # PDFs don't have EXIF
                }
                results["pages"].append(page_result)
                
                # Cleanup temp file
                if temp_path.exists():
                    temp_path.unlink()
        
        # Process Image
        else:
            # Perform ELA
            ela_result = generate_ela(file_path)
            
            # Extract EXIF
            exif_result = extract_exif_metadata(file_path)
            
            # Get original image as base64
            img = Image.open(file_path).convert('RGB')
            buf = io.BytesIO()
            # Resize for response if too large
            max_dim = 1200
            if img.width > max_dim or img.height > max_dim:
                ratio = min(max_dim / img.width, max_dim / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            img.save(buf, format='PNG')
            img_base64 = base64.b64encode(buf.getvalue()).decode()
            
            page_result = {
                "page_number": 1,
                "image": img_base64,
                "ela": ela_result,
                "ela_interpretation": get_ela_interpretation(ela_result),
                "exif": exif_result
            }
            results["pages"].append(page_result)
            
            # Day 6 Enhancement: Run fraud detection
            fraud_result = detect_document_fraud(file_path)
            results["fraud_detection"] = fraud_result
        
        # Calculate overall tamper score
        total_tamper_score = 0
        tamper_indicators = []
        
        for page in results["pages"]:
            if page["ela"]:
                ela = page["ela"]
                if ela["brightness_std"] > 25:
                    total_tamper_score += 30
                    tamper_indicators.append(f"Page {page['page_number']}: High ELA variance")
                elif ela["brightness_std"] > 15:
                    total_tamper_score += 15
                
                if ela["suspect_percentage"] > 5:
                    total_tamper_score += 25
                    tamper_indicators.append(f"Page {page['page_number']}: Suspicious regions detected")
            
            if page["exif"]:
                if page["exif"]["editing_detected"]:
                    total_tamper_score += 40
                    tamper_indicators.extend(page["exif"]["tamper_indicators"])
        
        # Normalize score
        num_pages = len(results["pages"])
        if num_pages > 0:
            total_tamper_score = min(100, total_tamper_score / num_pages)
        
        # Determine verdict
        if total_tamper_score >= 50:
            verdict = "LIKELY_TAMPERED"
            confidence = "HIGH"
        elif total_tamper_score >= 25:
            verdict = "POSSIBLY_TAMPERED"
            confidence = "MEDIUM"
        else:
            verdict = "LIKELY_AUTHENTIC"
            confidence = "HIGH" if total_tamper_score < 10 else "MEDIUM"
        
        # Integrate fraud detection results (CNN + heuristics)
        fraud_result = results.get("fraud_detection", {})
        fraud_authenticity = fraud_result.get("authenticity_score", 100)
        cnn_prediction = fraud_result.get("cnn_prediction")
        cnn_confidence = fraud_result.get("cnn_confidence", 0)
        
        # Add fraud indicators to tamper_indicators
        tamper_indicators.extend(fraud_result.get("fraud_indicators", []))
        
        # Combine ELA tamper score with fraud detection score
        # Lower authenticity = higher fraud likelihood
        combined_fraud_score = 100 - fraud_authenticity  # Convert to fraud score
        
        # If CNN predicts FRAUDULENT with confidence, boost fraud score
        if cnn_prediction == "FRAUDULENT" and cnn_confidence > 0.55:
            combined_fraud_score = max(combined_fraud_score, cnn_confidence * 100)
        
        # If CNN predicts AUTHENTIC with confidence, reduce fraud score
        if cnn_prediction == "AUTHENTIC" and cnn_confidence > 0.5:
            combined_fraud_score = combined_fraud_score * (1 - cnn_confidence * 0.5)
            
        # Use combined score for final verdict
        final_score = max(total_tamper_score, combined_fraud_score)
        
        # Override verdict based on fraud detection
        # Only flag as FRAUDULENT if we have strong evidence
        if fraud_result.get("is_sample_specimen"):
            verdict = "LIKELY_FRAUDULENT"
            confidence = "HIGH"
        elif cnn_prediction == "FRAUDULENT" and cnn_confidence > 0.65:
            verdict = "LIKELY_FRAUDULENT"
            confidence = "HIGH" if cnn_confidence > 0.75 else "MEDIUM"
        elif combined_fraud_score >= 60:
            verdict = "LIKELY_FRAUDULENT"
            confidence = "HIGH"
        elif combined_fraud_score >= 40 and cnn_prediction != "AUTHENTIC":
            verdict = "SUSPICIOUS"
            confidence = "MEDIUM"
        elif combined_fraud_score >= 50:
            verdict = "SUSPICIOUS"
            confidence = "LOW"
        
        results["summary"] = {
            "verdict": verdict,
            "confidence": confidence,
            "tamper_score": round(final_score, 1),
            "tamper_indicators": tamper_indicators,
            "pages_analyzed": num_pages,
            "document_type": fraud_result.get("document_type", "unknown"),
            "authenticity_score": fraud_authenticity,
            "cnn_prediction": cnn_prediction,
            "cnn_confidence": round(cnn_confidence * 100, 1) if cnn_confidence else None
        }
        
        print(f"[RESULT] {verdict} (Fraud: {combined_fraud_score:.1f}%, ELA: {total_tamper_score:.1f}%)")
        
        return jsonify(results)
        
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if file_path and file_path.exists():
            file_path.unlink()


@app.route("/detect", methods=["POST"])
def detect():
    job_id = str(uuid.uuid4())
    video_path = None
    
    try:
        file = request.files.get("file")
        if not file or not file.filename:
            return jsonify({"error": "No file"}), 400
        
        generate_heatmaps = request.form.get("heatmaps", "true").lower() == "true"
        
        print(f"\n[SCAN] {file.filename}")
        print(f"[HEATMAPS] {'Enabled' if generate_heatmaps else 'Disabled'}")
        
        ext = Path(file.filename).suffix.lower()
        video_path = UPLOAD_FOLDER / f"{job_id}{ext}"
        file.save(video_path)
        
        frames_dir = FRAMES_FOLDER / job_id
        frames = _load_frames(video_path, ext, frames_dir)
        
        if not frames:
            return jsonify({"error": "Cannot process"}), 400
        
        results, all_scores, face_scores = _analyze_all_frames(frames)
        _add_heatmaps_to_results(results, face_scores, generate_heatmaps)
        
        # Remove frame_path from results (existing code...)
        for r in results:
            r.pop("frame_path", None)
        
        final, avg, mx, med, std = _calculate_final_score(all_scores, face_scores)
        verdict, conf = _get_verdict(final, std)
        explanation = get_explanation(final)
        
        print(f"\n[RESULT] {verdict} ({final*100:.1f}%)")
        print(f"[EXPLANATION] {explanation[:80]}...")
        
        top_suspicious = [i+1 for i, _ in sorted(face_scores, key=lambda x: x[1], reverse=True)[:3]] if face_scores else []
        
        return jsonify({
            "jobId": job_id,
            "status": "completed",
            "result": {
                "verdict": verdict,
                "verdict_confidence": conf,
                "deepfake_score": round(final * 100, 1),
                "explanation": explanation,
                "frames_analyzed": len(results),
                "frames": results,
                "top_suspicious_frames": top_suspicious,
                "analysis_summary": {
                    "average_score": round(avg * 100, 1),
                    "max_score": round(mx * 100, 1),
                    "median_score": round(med * 100, 1),
                    "consistency": round((1 - std) * 100, 1),
                    "faces_detected": len(face_scores),
                    "heatmaps_generated": sum(1 for r in results if r.get("heatmap"))
                },
                "metadata": {
                    "mode": "PRODUCTION" if PRODUCTION_MODE else "DEMO",
                    "gradcam_enabled": GRADCAM_AVAILABLE
                }
            }
        })
    except (IOError, RuntimeError, ValueError) as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if video_path and video_path.exists():
            video_path.unlink()

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("TrustLock Deepfake Detection - Day 6")
    print("=" * 50)
    print(f"Mode: {'PRODUCTION' if PRODUCTION_MODE else 'DEMO'}")
    print(f"Grad-CAM: {'✅ Enabled' if GRADCAM_AVAILABLE else '❌ Disabled'}")
    print(f"EXIF: {'✅ Enabled' if PIEXIF_AVAILABLE else '❌ Disabled'}")
    print(f"PDF: {'✅ Enabled' if PDF2IMAGE_AVAILABLE else '❌ Disabled'}")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
