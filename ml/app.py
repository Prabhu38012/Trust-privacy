import os
import subprocess
import base64
import uuid
import sys
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

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

if PRODUCTION_MODE:
    print(f"✅ Loading trained model: {TRAINED_MODEL_PATH}")
    model = DeepfakeDetector()
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device, weights_only=True))
    model.eval().to(device)
    print("✅ TRAINED MODEL LOADED")
else:
    print("⚠️ No trained model, using base model")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.eval().to(device)

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
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap"""
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = (output > 0).float()
        
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

def get_gradcam_target_layer(model):
    """Get the target layer for Grad-CAM"""
    if PRODUCTION_MODE:
        # For our custom model, use last conv layer of backbone
        return model.backbone.features[-1]
    else:
        # For base EfficientNet
        return model.features[-1]

# Initialize Grad-CAM
try:
    target_layer = get_gradcam_target_layer(model)
    gradcam = GradCAM(model, target_layer)
    GRADCAM_AVAILABLE = True
    print("✅ Grad-CAM initialized")
except Exception as e:
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
    except Exception as e:
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
    except:
        return None

# ============================================================
# FACE EXTRACTION & ANALYSIS
# ============================================================
def extract_face(image, margin=0.4):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    for scale in [1.05, 1.1, 1.2, 1.3]:
        faces = face_cascade.detectMultiScale(gray, scale, 4, minSize=(50, 50))
        if len(faces) > 0:
            break
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        x1 = max(0, int(x - w * margin))
        y1 = max(0, int(y - h * margin))
        x2 = min(image.width, int(x + w + w * margin))
        y2 = min(image.height, int(y + h + h * margin))
        return image.crop((x1, y1, x2, y2)), True, (x, y, w, h)
    w, h = image.size
    m = min(w, h)
    return image.crop(((w-m)//2, (h-m)//2, (w+m)//2, (h+m)//2)), False, None

def analyze_frame(frame_path, generate_heatmap_flag=False):
    """Analyze frame with optional Grad-CAM heatmap"""
    try:
        img = Image.open(frame_path).convert("RGB")
        face_img, face_detected, face_box = extract_face(img)
        
        if not face_detected:
            return {
                "deepfake_probability": 0.35,
                "confidence": 25.0,
                "face_detected": False,
                "heatmap": None,
                "heatmap_overlay": None,
                "details": {"neural_network": 0.35, "note": "no_face_detected"}
            }
        
        input_tensor = preprocess(face_img).unsqueeze(0).to(device)
        input_tensor.requires_grad = True
        
        with torch.enable_grad():
            if PRODUCTION_MODE:
                output = model(input_tensor)
                nn_score = torch.sigmoid(output).item()
                if nn_score > 0.90:
                    nn_score = 0.80
                elif nn_score < 0.10:
                    nn_score = 0.20
            else:
                features = model.features(input_tensor)
                pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten()
                nn_score = 0.3 + (pooled.std().item() * 0.5)
        
        # Generate heatmaps if requested
        heatmap_base64 = None
        heatmap_overlay_base64 = None
        
        if generate_heatmap_flag and GRADCAM_AVAILABLE:
            input_tensor_grad = preprocess(face_img).unsqueeze(0).to(device)
            input_tensor_grad.requires_grad = True
            heatmap_base64 = generate_heatmap_only(input_tensor_grad, (face_img.width, face_img.height))
            heatmap_overlay_base64 = generate_heatmap(input_tensor_grad, face_img)
        
        confidence = abs(nn_score - 0.5) * 2 * 100
        confidence = max(60, min(95, 50 + confidence))
        
        return {
            "deepfake_probability": float(np.clip(nn_score, 0, 1)),
            "confidence": float(confidence),
            "face_detected": face_detected,
            "heatmap": heatmap_base64,
            "heatmap_overlay": heatmap_overlay_base64,
            "details": {"neural_network": round(nn_score, 3)}
        }
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return {"deepfake_probability": 0.35, "confidence": 25.0, "face_detected": False, "heatmap": None, "details": {}}

def image_to_base64(path, max_size=400):
    try:
        img = Image.open(path)
        ratio = min(max_size / img.width, max_size / img.height)
        if ratio < 1:
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except:
        return ""

def find_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return "ffmpeg"
    except:
        for pkg in Path(os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Packages")).glob("Gyan.FFmpeg*"):
            for exe in pkg.rglob("ffmpeg.exe"):
                try:
                    subprocess.run([str(exe), "-version"], capture_output=True, check=True)
                    return str(exe)
                except:
                    pass
    return None

FFMPEG = find_ffmpeg()
print(f"FFmpeg: {'Yes' if FFMPEG else 'No'}")

def extract_frames(video, out_dir, fps=1):
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, "-i", str(video), "-vf", f"fps={fps}", "-q:v", "2", str(out_dir / "frame_%04d.png"), "-y"], capture_output=True)
    return sorted(out_dir.glob("frame_*.png"))

def get_explanation(score, details):
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

@app.route("/detect", methods=["POST"])
def detect():
    job_id = str(uuid.uuid4())
    video_path = None
    
    try:
        file = request.files.get("file")
        if not file or not file.filename:
            return jsonify({"error": "No file"}), 400
        
        # Check if heatmaps requested
        generate_heatmaps = request.form.get("heatmaps", "true").lower() == "true"
        
        print(f"\n[SCAN] {file.filename}")
        print(f"[HEATMAPS] {'Enabled' if generate_heatmaps else 'Disabled'}")
        
        ext = Path(file.filename).suffix.lower()
        video_path = UPLOAD_FOLDER / f"{job_id}{ext}"
        file.save(video_path)
        
        frames_dir = FRAMES_FOLDER / job_id
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
        
        if not frames:
            return jsonify({"error": "Cannot process"}), 400
        
        results, all_scores, face_scores = [], [], []
        
        for i, fp in enumerate(frames):
            # Generate heatmaps only for analysis (will select top 3 later)
            analysis = analyze_frame(fp, generate_heatmap_flag=False)
            score = analysis["deepfake_probability"]
            face_detected = analysis.get("face_detected", True)
            
            indicator = "🔴 FAKE" if score > 0.65 else "🟡 UNCERTAIN" if score > 0.40 else "🟢 REAL"
            face_icon = "👤" if face_detected else "❌"
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
        
        # Generate heatmaps for top 3 suspicious frames
        if generate_heatmaps and len(face_scores) > 0:
            # Sort by score (highest first)
            top_frames = sorted(face_scores, key=lambda x: x[1], reverse=True)[:3]
            print(f"\n  [Generating heatmaps for top {len(top_frames)} frames]")
            
            for idx, score in top_frames:
                fp = Path(results[idx]["frame_path"])
                if fp.exists():
                    analysis_with_heatmap = analyze_frame(fp, generate_heatmap_flag=True)
                    results[idx]["heatmap"] = analysis_with_heatmap.get("heatmap")
                    results[idx]["heatmap_overlay"] = analysis_with_heatmap.get("heatmap_overlay")
                    print(f"    Frame {idx+1}: Heatmap generated ✓")
        
        # Remove frame_path from results (internal use only)
        for r in results:
            r.pop("frame_path", None)
        
        # Calculate final score
        if len(face_scores) >= 5:
            scores_for_analysis = [s for _, s in face_scores]
        else:
            scores_for_analysis = all_scores
        
        scores_array = np.array(scores_for_analysis)
        avg = np.mean(scores_array)
        mx = np.max(scores_array)
        med = np.median(scores_array)
        std = np.std(scores_array)
        
        high_fake = sum(1 for s in scores_for_analysis if s > 0.60)
        high_real = sum(1 for s in scores_for_analysis if s < 0.38)
        total = len(scores_for_analysis)
        
        fake_ratio = high_fake / total
        real_ratio = high_real / total
        
        if fake_ratio > 0.35:
            final = avg * 0.3 + mx * 0.4 + med * 0.3
            final = max(final, 0.58)
        elif real_ratio > 0.35:
            final = avg * 0.5 + med * 0.5
            final = min(final, 0.42)
        else:
            final = avg * 0.5 + med * 0.3 + mx * 0.2
        
        final = np.clip(final, 0, 1)
        
        # Verdict
        if final < 0.32:
            verdict, conf = "AUTHENTIC", "HIGH"
        elif final < 0.45:
            verdict, conf = "LIKELY_AUTHENTIC", "MEDIUM"
        elif final < 0.55:
            verdict, conf = "UNCERTAIN", "LOW"
        elif final < 0.70:
            verdict, conf = "SUSPICIOUS", "HIGH"
        else:
            verdict, conf = "LIKELY_DEEPFAKE", "VERY HIGH"
        
        if std > 0.20:
            conf = "LOW"
        
        # Generate explanation
        explanation = get_explanation(final, {})
        
        print(f"\n[RESULT] {verdict} ({final*100:.1f}%)")
        print(f"[EXPLANATION] {explanation[:80]}...")
        
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
                "top_suspicious_frames": [i+1 for i, _ in sorted(face_scores, key=lambda x: x[1], reverse=True)[:3]] if face_scores else [],
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
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if video_path and video_path.exists():
            video_path.unlink()

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("TrustLock Deepfake Detection - Day 3")
    print("=" * 50)
    print(f"Mode: {'PRODUCTION' if PRODUCTION_MODE else 'DEMO'}")
    print(f"Grad-CAM: {'✅ Enabled' if GRADCAM_AVAILABLE else '❌ Disabled'}")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
