# Quick Start Guide: HoverNet + TCAV Prototype

This guide gets you started immediately with downloading data and running the prototype.

## Prerequisites

- macOS/Linux with NVIDIA GPU (16GB+ VRAM recommended)
- Python 3.8+
- 500GB+ storage space
- TCIA account (free registration)

---

## Step 1: Environment Setup (15 minutes)

```bash
# Navigate to HoverNet directory
cd /Users/rafik.salama/Codebase/HoverNet

# Create conda environment
conda create -n hovernet_tcav python=3.8 -y
conda activate hovernet_tcav

# Install PyTorch with CUDA support
conda install pytorch torchvision cudatoolkit=11.8 -c pytorch -y

# Install TensorFlow
pip install tensorflow==2.13.0

# Install core dependencies
pip install numpy scipy scikit-learn pandas matplotlib seaborn plotly

# Install image processing libraries
pip install openslide-python opencv-python Pillow albumentations

# Install stain normalization
pip install staintools

# Install histopathology tools
pip install histolab
```

---

## Step 2: Download Sample Dataset (30 minutes)

### Option A: HER2-TUMOR-ROIS Dataset (Recommended - Has pCR Labels!)

**Dataset Info:**
- 36 HER2+ breast cancer patients
- Pre-treatment H&E slides
- Complete pCR response labels
- Source: The Cancer Imaging Archive (TCIA)

**Download Instructions:**

1. **Register for TCIA Account:**
   - Go to: https://www.cancerimagingarchive.net/
   - Click "Register" (top right)
   - Fill out form (free, instant approval)

2. **Install NBIA Data Retriever:**
   ```bash
   # For macOS
   wget https://cbiit-download.nci.nih.gov/nbia/releases/ForTCIA/NBIADataRetriever_4.4/nbia-data-retriever-4.4.dmg
   open nbia-data-retriever-4.4.dmg
   # Drag to Applications folder
   ```

3. **Download HER2-TUMOR-ROIS Collection:**
   - Go to: https://www.cancerimagingarchive.net/collection/her2-tumor-rois/
   - Click "Download" tab
   - Click "Add all data to cart"
   - Click "Download" and save manifest file: `her2-tumor-rois-manifest.tcia`

4. **Download Data:**
   ```bash
   # Create data directory
   mkdir -p data/raw/HER2-TUMOR-ROIS

   # Open NBIA Data Retriever application
   # File → Open Manifest → Select her2-tumor-rois-manifest.tcia
   # Set destination: /Users/rafik.salama/Codebase/HoverNet/data/raw/HER2-TUMOR-ROIS
   # Click "Start Download"
   ```

   **Expected Download:**
   - Size: ~10-15 GB
   - Files: 36 H&E whole-slide images (.svs format)
   - Clinical data CSV with pCR labels

### Option B: Quick Test with Sample Images

If you want to test immediately without waiting for full download:

```bash
# Download a few sample breast H&E images
mkdir -p data/raw/samples

# Sample 1: Normal breast tissue
wget https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs \
  -O data/raw/samples/sample_1.svs

# Sample 2: Breast carcinoma
wget https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs \
  -O data/raw/samples/sample_2.svs
```

---

## Step 3: Install HoverNet (15 minutes)

```bash
cd /Users/rafik.salama/Codebase/HoverNet

# Clone HoverNet repository
git clone https://github.com/vqdang/hover_net.git
cd hover_net

# Install HoverNet dependencies
pip install -r requirements.txt

# Install additional required packages
pip install scikit-image imageio

# Download pre-trained weights
mkdir -p pretrained
cd pretrained

# Download HoverNet model trained on PanNuke dataset
wget https://drive.google.com/uc?export=download&id=1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq -O hovernet_pannuke.tar

# Extract
tar -xvf hovernet_pannuke.tar

cd ../..
```

**Verify Installation:**
```bash
# Test HoverNet
python hover_net/run_infer.py --help
```

---

## Step 4: Install TCAV (10 minutes)

```bash
cd /Users/rafik.salama/Codebase/HoverNet

# Clone TCAV repository
git clone https://github.com/tensorflow/tcav.git
cd tcav

# Install TCAV
pip install -e .

cd ..
```

**Verify Installation:**
```bash
python -c "import tcav; print('TCAV installed successfully!')"
```

---

## Step 5: Run Your First Analysis (30 minutes)

### 5.1 Create Quick Test Script

```bash
# Create scripts directory
mkdir -p scripts
```

Create `scripts/test_hovernet.py`:

```python
#!/usr/bin/env python3
"""Test HoverNet on a sample slide"""

import sys
sys.path.append('hover_net')

import numpy as np
import cv2
from pathlib import Path
import openslide

def test_hovernet_on_sample():
    """Run HoverNet inference on sample slide"""

    # Path to sample slide
    slide_path = "data/raw/samples/sample_1.svs"

    if not Path(slide_path).exists():
        print(f"Error: {slide_path} not found!")
        print("Please download sample data first.")
        return

    print(f"Loading slide: {slide_path}")
    slide = openslide.OpenSlide(slide_path)

    # Extract a region at 20x magnification
    # Location (x, y) in level 0 coordinates
    location = (10000, 10000)
    size = (512, 512)

    print(f"Extracting region at {location}, size {size}")
    region = slide.read_region(location, 0, size)
    region_rgb = np.array(region.convert('RGB'))

    # Save for inspection
    cv2.imwrite("results/test_region.png", cv2.cvtColor(region_rgb, cv2.COLOR_RGB2BGR))
    print("Saved test region to: results/test_region.png")

    # TODO: Run HoverNet inference
    # This will be implemented after verifying the region looks good
    print("\nNext steps:")
    print("1. Check results/test_region.png to verify tissue quality")
    print("2. Run HoverNet segmentation on this region")
    print("3. Visualize nuclei segmentation results")

    slide.close()

if __name__ == "__main__":
    # Create results directory
    Path("results").mkdir(exist_ok=True)

    test_hovernet_on_sample()
```

### 5.2 Run Test

```bash
# Make script executable
chmod +x scripts/test_hovernet.py

# Run test
python scripts/test_hovernet.py
```

**Expected Output:**
```
Loading slide: data/raw/samples/sample_1.svs
Extracting region at (10000, 10000), size (512, 512)
Saved test region to: results/test_region.png

Next steps:
1. Check results/test_region.png to verify tissue quality
2. Run HoverNet segmentation on this region
3. Visualize nuclei segmentation results
```

### 5.3 Inspect Results

```bash
# Open the extracted region
open results/test_region.png
```

You should see a 512x512 pixel H&E tissue image!

---

## Step 6: Run HoverNet Segmentation

Create `scripts/run_hovernet_segmentation.py`:

```python
#!/usr/bin/env python3
"""Run HoverNet nuclei segmentation"""

import sys
sys.path.append('hover_net')

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def segment_nuclei(input_image_path):
    """
    Segment nuclei using HoverNet

    For now, we'll use a simplified approach.
    Full HoverNet integration will come in Phase 1.
    """
    print(f"Loading image: {input_image_path}")
    img = cv2.imread(str(input_image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Placeholder: Simple nuclear detection using color thresholding
    # This will be replaced with HoverNet inference

    # Convert to grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Simple thresholding (nuclei are darker)
    _, nuclei_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Find contours (nuclei candidates)
    contours, _ = cv2.findContours(nuclei_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by size (nuclei are typically 5-20 pixels in diameter at this resolution)
    valid_nuclei = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 500:  # Adjust based on magnification
            valid_nuclei.append(cnt)

    print(f"Detected {len(valid_nuclei)} nuclei candidates")

    # Visualize
    result_img = img_rgb.copy()
    cv2.drawContours(result_img, valid_nuclei, -1, (0, 255, 0), 1)

    # Save result
    output_path = "results/nuclei_segmentation.png"
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(result_img)
    plt.title(f'Nuclei Detected: {len(valid_nuclei)}')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved segmentation result to: {output_path}")

    return len(valid_nuclei)

if __name__ == "__main__":
    input_path = "results/test_region.png"

    if not Path(input_path).exists():
        print(f"Error: {input_path} not found!")
        print("Run test_hovernet.py first to extract a test region.")
    else:
        segment_nuclei(input_path)
```

```bash
# Run segmentation
python scripts/run_hovernet_segmentation.py

# View results
open results/nuclei_segmentation.png
```

---

## Step 7: Verify Installation Checklist

```bash
# Create verification script
cat > scripts/verify_installation.sh << 'EOF'
#!/bin/bash

echo "=== HoverNet + TCAV Installation Verification ==="
echo ""

# Check Python environment
echo "✓ Checking Python version..."
python --version

# Check PyTorch
echo "✓ Checking PyTorch..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"

# Check TensorFlow
echo "✓ Checking TensorFlow..."
python -c "import tensorflow as tf; print(f'  TensorFlow: {tf.__version__}')"

# Check OpenSlide
echo "✓ Checking OpenSlide..."
python -c "import openslide; print(f'  OpenSlide: {openslide.__version__}')"

# Check TCAV
echo "✓ Checking TCAV..."
python -c "import tcav; print('  TCAV: Installed')"

# Check HoverNet
echo "✓ Checking HoverNet..."
if [ -d "hover_net" ]; then
    echo "  HoverNet: Installed"
else
    echo "  HoverNet: NOT FOUND"
fi

# Check data directories
echo "✓ Checking data directories..."
if [ -d "data/raw" ]; then
    echo "  data/raw: Exists"
    echo "  Files: $(find data/raw -name '*.svs' 2>/dev/null | wc -l) .svs files found"
else
    echo "  data/raw: NOT FOUND"
fi

echo ""
echo "=== Verification Complete ==="
EOF

chmod +x scripts/verify_installation.sh
./scripts/verify_installation.sh
```

---

## Next Steps

### Immediate (Today):
1. ✅ Complete environment setup
2. ✅ Download HER2-TUMOR-ROIS dataset
3. ✅ Run test scripts to verify installation

### This Week:
4. **Process full dataset with HoverNet**
   - Segment nuclei in all 36 slides
   - Extract morphological features
   - Compute TIL density, tumor cellularity

5. **Define pathological concepts**
   - Collect example patches for:
     - High TILs
     - Geographic necrosis
     - Viable tumor
     - Fibrosis

6. **Set up TCAV analysis**
   - Train concept activation vectors
   - Compute TCAV scores
   - Visualize concept importance

### Next 2 Weeks:
7. Build MIL classifier for pCR prediction
8. Integrate interpretability (attention + concepts)
9. Run validation experiments
10. Generate pathologist review reports

---

## Troubleshooting

### Issue: OpenSlide not found
```bash
# macOS
brew install openslide

# Linux
sudo apt-get install openslide-tools
```

### Issue: CUDA not available
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Download speed slow
- Use university VPN or high-bandwidth connection
- Download during off-peak hours
- Consider downloading in batches

### Issue: Out of memory during processing
- Reduce tile size (use 256x256 instead of 512x512)
- Process slides sequentially instead of in parallel
- Use CPU for inference if GPU memory insufficient

---

## Getting Help

- **HoverNet Issues:** https://github.com/vqdang/hover_net/issues
- **TCAV Issues:** https://github.com/tensorflow/tcav/issues
- **TCIA Data:** https://www.cancerimagingarchive.net/support/
- **General Questions:** rafik.salama@codebase

---

## Summary

You should now have:
- ✅ Working conda environment with all dependencies
- ✅ HoverNet installed with pre-trained weights
- ✅ TCAV framework installed
- ✅ Sample H&E slides downloaded
- ✅ Test scripts to verify everything works

**Ready to proceed with full implementation!**

See [PROTOTYPE_IMPLEMENTATION_PLAN.md](PROTOTYPE_IMPLEMENTATION_PLAN.md) for detailed technical specifications.