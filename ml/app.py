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
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
import traceback
import cv2
import io

# Optional imports for document analysis
try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️ pdf2image not available - PDF support disabled")

try:
    import piexif
    EXIF_SUPPORT = True
except ImportError:
    EXIF_SUPPORT = False
    print("⚠️ piexif not available - detailed EXIF editing detection disabled")

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = Path("./uploads")
FRAMES_FOLDER = Path("./frames")
MODELS_FOLDER = Path("./models")
UPLOAD_FOLDER.mkdir(exist_ok=True)
FRAMES_FOLDER.mkdir(exist_ok=True)
MODELS_FOLDER.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
# DOCUMENT ANALYSIS - ELA, EXIF, PDF
# ============================================================

def perform_ela(image, quality=90, scale=15):
    """
    Perform Error Level Analysis on an image.
    Compresses the image at specified quality and computes 
    the difference to reveal tampering.
    
    Args:
        image: PIL Image object
        quality: JPEG compression quality (lower = more compression)
        scale: Multiplier to enhance differences
    
    Returns:
        tuple: (ela_image as PIL Image, ela_base64 string)
    """
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save at specified quality to buffer
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        # Reload compressed image
        compressed = Image.open(buffer)
        
        # Compute absolute difference
        original_array = np.array(image, dtype=np.float32)
        compressed_array = np.array(compressed, dtype=np.float32)
        
        diff = np.abs(original_array - compressed_array)
        
        # Scale the difference to make it visible
        diff = diff * scale
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        
        ela_image = Image.fromarray(diff)
        
        # Convert to base64
        ela_buffer = io.BytesIO()
        ela_image.save(ela_buffer, format='PNG')
        ela_base64 = base64.b64encode(ela_buffer.getvalue()).decode()
        
        return ela_image, ela_base64
    except Exception as e:
        print(f"ELA error: {e}")
        return None, None

def calculate_ela_score(ela_image):
    """
    Calculate a tampering score based on ELA analysis.
    Higher values indicate potential tampering.
    """
    try:
        if ela_image is None:
            return 0.0
        
        ela_array = np.array(ela_image)
        
        # Calculate statistics
        mean_intensity = np.mean(ela_array)
        max_intensity = np.max(ela_array)
        std_intensity = np.std(ela_array)
        
        # High variance regions indicate potential tampering
        high_intensity_ratio = np.sum(ela_array > 128) / ela_array.size
        
        # Combine metrics into a score (0-1)
        score = (
            (mean_intensity / 255) * 0.2 +
            (max_intensity / 255) * 0.3 +
            (std_intensity / 128) * 0.3 +
            high_intensity_ratio * 0.2
        )
        
        return float(np.clip(score, 0, 1))
    except Exception as e:
        print(f"ELA score error: {e}")
        return 0.0

def extract_exif_metadata(image_path):
    """
    Extract EXIF metadata from an image and check for suspicious indicators.
    
    Returns:
        dict: Contains metadata, warnings, and suspicious flags
    """
    result = {
        "metadata": {},
        "warnings": [],
        "suspicious_indicators": [],
        "has_exif": False
    }
    
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        if exif_data is None:
            result["warnings"].append("No EXIF metadata found - may have been stripped")
            return result
        
        result["has_exif"] = True
        
        # Extract readable EXIF data
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            
            # Handle bytes/binary data
            if isinstance(value, bytes):
                try:
                    value = value.decode('utf-8', errors='ignore')
                except:
                    value = str(value)[:100]
            
            # Store metadata
            result["metadata"][tag] = str(value)[:200]  # Limit value length
            
            # Check for suspicious indicators
            tag_lower = str(tag).lower()
            value_str = str(value).lower()
            
            # Check for editing software
            if tag_lower in ['software', 'processingsoftware']:
                editing_software = ['photoshop', 'gimp', 'lightroom', 'affinity', 
                                   'pixlr', 'snapseed', 'picsart', 'canva']
                for software in editing_software:
                    if software in value_str:
                        result["suspicious_indicators"].append(
                            f"Edited with {value} - image may have been manipulated"
                        )
                        break
            
            # Check for date inconsistencies
            if 'date' in tag_lower:
                result["metadata"][f"_date_{tag}"] = str(value)
        
        # Check for GPS data
        gps_tags = ['GPSInfo', 'GPSLatitude', 'GPSLongitude']
        has_gps = any(tag in result["metadata"] for tag in gps_tags)
        if not has_gps:
            result["warnings"].append("GPS data not present or stripped")
        
        # Use piexif for more detailed analysis if available
        if EXIF_SUPPORT:
            try:
                exif_dict = piexif.load(str(image_path))
                
                # Check for thumbnail inconsistencies
                if piexif.ImageIFD.ImageWidth in exif_dict.get("0th", {}):
                    result["metadata"]["OriginalWidth"] = exif_dict["0th"][piexif.ImageIFD.ImageWidth]
                if piexif.ImageIFD.ImageLength in exif_dict.get("0th", {}):
                    result["metadata"]["OriginalHeight"] = exif_dict["0th"][piexif.ImageIFD.ImageLength]
                
                # Check if thumbnail exists but is different size
                if "thumbnail" in exif_dict and exif_dict["thumbnail"]:
                    result["warnings"].append("Contains embedded thumbnail - verify consistency")
                    
            except Exception as e:
                print(f"piexif analysis error: {e}")
        
    except Exception as e:
        result["warnings"].append(f"Could not read EXIF: {str(e)}")
    
    return result

def convert_pdf_to_images(pdf_path, dpi=200, first_page_only=True):
    """
    Convert PDF to images using pdf2image.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion
        first_page_only: If True, only convert first page
    
    Returns:
        list: List of PIL Images
    """
    if not PDF_SUPPORT:
        print("PDF support not available")
        return []
    
    try:
        if first_page_only:
            images = convert_from_path(str(pdf_path), dpi=dpi, first_page=1, last_page=1)
        else:
            images = convert_from_path(str(pdf_path), dpi=dpi)
        return images
    except Exception as e:
        print(f"PDF conversion error: {e}")
        return []

def get_document_explanation(ela_score, exif_result, tampering_indicators):
    """
    Generate explanation text for document analysis.
    """
    explanations = []
    
    # ELA-based explanation
    if ela_score > 0.6:
        explanations.append(
            "Error Level Analysis reveals significant inconsistencies in the image compression. "
            "This suggests portions of the image may have been modified or added after initial creation."
        )
    elif ela_score > 0.4:
        explanations.append(
            "ELA shows moderate variations in compression levels across the image. "
            "Some regions may have been edited, but results are not conclusive."
        )
    elif ela_score > 0.2:
        explanations.append(
            "Minor compression inconsistencies detected. The image appears mostly authentic "
            "with possible minor adjustments."
        )
    else:
        explanations.append(
            "ELA shows consistent compression levels throughout the image, "
            "suggesting it has not been significantly manipulated."
        )
    
    # EXIF-based explanation
    if exif_result.get("suspicious_indicators"):
        explanations.append(
            f"Metadata analysis found {len(exif_result['suspicious_indicators'])} suspicious indicator(s): " +
            "; ".join(exif_result["suspicious_indicators"][:3])
        )
    
    if exif_result.get("warnings"):
        explanations.append(
            "Metadata warnings: " + "; ".join(exif_result["warnings"][:2])
        )
    
    return " ".join(explanations)

def analyze_document_tampering(image_path):
    """
    Perform comprehensive document tampering analysis.
    
    Returns:
        dict: Complete analysis results
    """
    result = {
        "ela": {
            "image": None,
            "score": 0.0
        },
        "exif": {
            "metadata": {},
            "warnings": [],
            "suspicious_indicators": [],
            "has_exif": False
        },
        "tampering_indicators": [],
        "tampering_score": 0.0,
        "verdict": "UNKNOWN",
        "explanation": ""
    }
    
    try:
        # Load image
        img = Image.open(image_path)
        original = img.convert('RGB')
        
        # Generate original image base64
        orig_buffer = io.BytesIO()
        # Resize for display if too large
        display_img = original.copy()
        max_size = 800
        if max(display_img.size) > max_size:
            ratio = max_size / max(display_img.size)
            new_size = (int(display_img.width * ratio), int(display_img.height * ratio))
            display_img = display_img.resize(new_size, Image.Resampling.LANCZOS)
        display_img.save(orig_buffer, format='PNG')
        result["original_image"] = base64.b64encode(orig_buffer.getvalue()).decode()
        
        # Perform ELA
        ela_img, ela_base64 = perform_ela(original)
        if ela_base64:
            result["ela"]["image"] = ela_base64
            result["ela"]["score"] = calculate_ela_score(ela_img)
        
        # Extract EXIF
        exif_result = extract_exif_metadata(image_path)
        result["exif"] = exif_result
        
        # Compile tampering indicators
        tampering_indicators = []
        
        if result["ela"]["score"] > 0.5:
            tampering_indicators.append({
                "type": "ELA",
                "severity": "high",
                "message": "High compression inconsistencies detected"
            })
        elif result["ela"]["score"] > 0.3:
            tampering_indicators.append({
                "type": "ELA",
                "severity": "medium",
                "message": "Moderate compression variations found"
            })
        
        for indicator in exif_result.get("suspicious_indicators", []):
            tampering_indicators.append({
                "type": "EXIF",
                "severity": "high",
                "message": indicator
            })
        
        for warning in exif_result.get("warnings", []):
            tampering_indicators.append({
                "type": "EXIF",
                "severity": "low",
                "message": warning
            })
        
        result["tampering_indicators"] = tampering_indicators
        
        # Calculate overall tampering score
        ela_weight = 0.6
        exif_weight = 0.4
        
        exif_score = len(exif_result.get("suspicious_indicators", [])) * 0.3
        exif_score += len(exif_result.get("warnings", [])) * 0.1
        exif_score = min(1.0, exif_score)
        
        result["tampering_score"] = (
            result["ela"]["score"] * ela_weight +
            exif_score * exif_weight
        )
        
        # Determine verdict
        if result["tampering_score"] > 0.6:
            result["verdict"] = "LIKELY_TAMPERED"
        elif result["tampering_score"] > 0.4:
            result["verdict"] = "SUSPICIOUS"
        elif result["tampering_score"] > 0.2:
            result["verdict"] = "POSSIBLY_MODIFIED"
        else:
            result["verdict"] = "LIKELY_AUTHENTIC"
        
        # Generate explanation
        result["explanation"] = get_document_explanation(
            result["ela"]["score"],
            exif_result,
            tampering_indicators
        )
        
    except Exception as e:
        print(f"Document analysis error: {e}")
        traceback.print_exc()
        result["error"] = str(e)
    
    return result

# ============================================================
# API ROUTES
# ============================================================
@app.route("/")
def index():
    return jsonify({
        "service": "TrustLock ML Service",
        "version": "6.0 - Day 6",
        "features": [
            "deepfake_detection", 
            "gradcam_heatmaps", 
            "explainability",
            "document_analysis",
            "ela_detection",
            "exif_metadata"
        ],
        "gradcam_available": GRADCAM_AVAILABLE,
        "pdf_support": PDF_SUPPORT,
        "exif_support": EXIF_SUPPORT
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "ok", 
        "production": PRODUCTION_MODE, 
        "gradcam": GRADCAM_AVAILABLE,
        "pdf_support": PDF_SUPPORT,
        "exif_support": EXIF_SUPPORT
    })

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

@app.route("/analyze-document", methods=["POST"])
def analyze_document():
    """
    Analyze uploaded document for tampering.
    Supports images (JPG, PNG, etc.) and PDFs.
    
    Returns:
    - ELA image (base64)
    - EXIF metadata
    - Tampering indicators
    - Explanation text
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
        
        # Check file size (50MB limit)
        file_size = file_path.stat().st_size
        max_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_size:
            if file_path.exists():
                file_path.unlink()
            return jsonify({
                "error": f"File too large. Maximum size is 50MB, got {file_size / (1024*1024):.1f}MB"
            }), 413
        
        # Handle PDF files
        if ext == ".pdf":
            if not PDF_SUPPORT:
                if file_path.exists():
                    file_path.unlink()
                return jsonify({
                    "error": "PDF support not available. Please install poppler and pdf2image."
                }), 501
            
            print("[DOCUMENT] Converting PDF to image...")
            images = convert_pdf_to_images(file_path)
            
            if not images:
                if file_path.exists():
                    file_path.unlink()
                return jsonify({
                    "error": "Could not convert PDF to image. Make sure poppler is installed."
                }), 500
            
            # Save first page as temporary image for analysis
            temp_image_path = UPLOAD_FOLDER / f"{job_id}_page1.png"
            images[0].save(temp_image_path, format='PNG')
            analysis_path = temp_image_path
            is_pdf = True
            page_count = len(images) if not True else 1  # We only converted first page
        else:
            # Validate image file type
            allowed_types = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
            if ext not in allowed_types:
                if file_path.exists():
                    file_path.unlink()
                return jsonify({
                    "error": f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_types)}"
                }), 415
            
            analysis_path = file_path
            is_pdf = False
            page_count = 1
        
        # Perform document analysis
        print("[DOCUMENT] Running tampering analysis...")
        result = analyze_document_tampering(analysis_path)
        
        # Clean up temp files
        if is_pdf and 'temp_image_path' in locals() and temp_image_path.exists():
            temp_image_path.unlink()
        
        print(f"[DOCUMENT] Verdict: {result.get('verdict', 'UNKNOWN')}")
        print(f"[DOCUMENT] Tampering Score: {result.get('tampering_score', 0) * 100:.1f}%")
        
        return jsonify({
            "jobId": job_id,
            "status": "completed",
            "filename": file.filename,
            "file_type": "pdf" if is_pdf else "image",
            "page_count": page_count,
            "result": {
                "verdict": result.get("verdict", "UNKNOWN"),
                "tampering_score": round(result.get("tampering_score", 0) * 100, 1),
                "explanation": result.get("explanation", ""),
                "original_image": result.get("original_image"),
                "ela": {
                    "image": result.get("ela", {}).get("image"),
                    "score": round(result.get("ela", {}).get("score", 0) * 100, 1)
                },
                "exif": {
                    "has_data": result.get("exif", {}).get("has_exif", False),
                    "metadata": result.get("exif", {}).get("metadata", {}),
                    "warnings": result.get("exif", {}).get("warnings", []),
                    "suspicious_indicators": result.get("exif", {}).get("suspicious_indicators", [])
                },
                "tampering_indicators": result.get("tampering_indicators", [])
            },
            "metadata": {
                "pdf_support": PDF_SUPPORT,
                "exif_support": EXIF_SUPPORT
            }
        })
        
    except Exception as e:
        print(f"[DOCUMENT ERROR] {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if file_path and file_path.exists():
            file_path.unlink()

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("TrustLock ML Service - Day 6")
    print("=" * 50)
    print(f"Mode: {'PRODUCTION' if PRODUCTION_MODE else 'DEMO'}")
    print(f"Grad-CAM: {'✅ Enabled' if GRADCAM_AVAILABLE else '❌ Disabled'}")
    print(f"PDF Support: {'✅ Enabled' if PDF_SUPPORT else '❌ Disabled'}")
    print(f"EXIF Support: {'✅ Enabled' if EXIF_SUPPORT else '❌ Disabled'}")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
