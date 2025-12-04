# ML Models

Model files are not included in the repository due to size limits.

## Setup

1. **Generate training data:**
   ```bash
   cd ../train
   python download_data.py
   ```

2. **Train the model:**
   ```bash
   python train.py
   ```

3. The trained model will be saved to `deepfake_detector.pth`

## Alternative: Download pre-trained weights

Place any `.pth` model file in this directory.
