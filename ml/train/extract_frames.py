# pylint: disable=no-member
"""Extract face frames from videos for training"""

import os
import shutil
import argparse
import subprocess
from pathlib import Path
import cv2
from tqdm import tqdm

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def find_ffmpeg():
    """Find ffmpeg"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return "ffmpeg"
    except (FileNotFoundError, subprocess.CalledProcessError):
        for pkg in Path(os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Packages")).glob("Gyan.FFmpeg*"):
            for exe in pkg.rglob("ffmpeg.exe"):
                return str(exe)
    return None

FFMPEG = find_ffmpeg()

def extract_frames_from_video(video_path, output_dir, fps=1, max_frames=30):
    """Extract frames from video"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if FFMPEG:
        cmd = [FFMPEG, "-i", str(video_path), "-vf", f"fps={fps}", 
               "-q:v", "2", str(output_dir / "frame_%04d.jpg"), "-y"]
        subprocess.run(cmd, capture_output=True, check=False)
    else:
        # Use OpenCV
        cap = cv2.VideoCapture(str(video_path))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = int(fps_video / fps)
        
        count = 0
        saved = 0
        while saved < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                cv2.imwrite(str(output_dir / f"frame_{saved:04d}.jpg"), frame)
                saved += 1
            count += 1
        cap.release()
    
    return list(output_dir.glob("*.jpg"))

def extract_face(img_path, output_path, margin=0.3):
    """Extract face from image and save"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            m = int(w * margin)
            x1, y1 = max(0, x - m), max(0, y - m)
            x2, y2 = min(img.shape[1], x + w + m), min(img.shape[0], y + h + m)
            
            face = img[y1:y2, x1:x2]
            face = cv2.resize(face, (256, 256))
            cv2.imwrite(str(output_path), face)
            return True
    except Exception:  # cv2.error doesn't inherit from Exception properly
        pass
    return False

def process_videos(input_dir, output_dir, label, fps=1, max_per_video=10):
    """Process all videos in directory"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    videos = [f for f in input_dir.glob("**/*") if f.suffix.lower() in video_extensions]
    
    print(f"Found {len(videos)} videos in {input_dir}")
    
    count = 0
    for video_path in tqdm(videos, desc=f"Processing {label}"):
        temp_dir = Path("./temp_frames")
        frames = extract_frames_from_video(video_path, temp_dir, fps=fps, max_frames=max_per_video)
        
        for frame_path in frames[:max_per_video]:
            output_path = output_dir / f"{label}_{count:06d}.jpg"
            if extract_face(frame_path, output_path):
                count += 1
        
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    print(f"✅ Extracted {count} face images to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract faces from videos")
    parser.add_argument("--real_videos", type=str, help="Directory with real videos")
    parser.add_argument("--fake_videos", type=str, help="Directory with fake videos")
    parser.add_argument("--output", type=str, default="./data", help="Output directory")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract")
    parser.add_argument("--max_per_video", type=int, default=10, help="Max frames per video")
    args = parser.parse_args()
    
    if args.real_videos:
        process_videos(args.real_videos, Path(args.output) / "real", "real", args.fps, args.max_per_video)
    
    if args.fake_videos:
        process_videos(args.fake_videos, Path(args.output) / "fake", "fake", args.fps, args.max_per_video)
    
    if not args.real_videos and not args.fake_videos:
        print("Usage:")
        print("  python extract_frames.py --real_videos ./real_vids --fake_videos ./fake_vids")

if __name__ == "__main__":
    main()
