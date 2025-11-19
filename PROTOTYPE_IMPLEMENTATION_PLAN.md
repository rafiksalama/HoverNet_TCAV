# HoverNet + TCAV Prototype Implementation Plan
## Interpretability for H&E-Based pCR Response Prediction Models

**Version:** 1.0
**Date:** 2025-11-19
**Objective:** Build an interpretable AI system for predicting pathological complete response (pCR) from H&E whole-slide images using HoverNet for nuclei segmentation and TCAV for concept-based explanations.

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Environment Setup](#environment-setup)
4. [Dataset Acquisition](#dataset-acquisition)
5. [Phase 1: HoverNet Implementation](#phase-1-hovernet-implementation)
6. [Phase 2: TCAV Integration](#phase-2-tcav-integration)
7. [Phase 3: MIL Model & Interpretability](#phase-3-mil-model--interpretability)
8. [Validation & Testing](#validation--testing)
9. [Timeline & Milestones](#timeline--milestones)

---

## Executive Summary

### Problem Statement
Develop an interpretable whole-slide image (WSI) analysis system that:
- Predicts pathological complete response (pCR) to neoadjuvant chemotherapy
- Provides human-understandable explanations of predictions
- Links model decisions to histopathological concepts
- Ensures "right for the right reasons" via robust validation

### Solution Components
1. **HoverNet**: Nuclei segmentation and classification to extract morphological features
2. **TCAV**: Concept-based explanations to quantify importance of pathological patterns
3. **MIL Model**: Attention-based multiple instance learning for WSI-level prediction
4. **Validation Framework**: Faithfulness, stability, and clinical plausibility tests

### Expected Outcomes
- Interpretable pCR prediction model with AUC > 0.80
- Concept attribution scores for key pathological features (TILs, necrosis, tumor cellularity)
- Visual attention heatmaps linked to semantic concepts
- Validated explanations approved by pathologists

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: H&E WSI (.svs/.tiff)              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              PREPROCESSING PIPELINE                          │
│  • Stain Normalization (Macenko)                            │
│  • Quality Control (HistoQC)                                │
│  • Tile Extraction (256x256 or 512x512)                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴────────────────┐
          │                                │
          ▼                                ▼
┌──────────────────────┐        ┌──────────────────────┐
│   HoverNet Module    │        │  Feature Extractor   │
│  • Nuclei Segment.   │        │  (Foundation Model)  │
│  • Cell Classification│        │  • CONCH/TITAN      │
│  • Morphometrics     │        │  • Patch Embeddings  │
└──────────┬───────────┘        └──────────┬───────────┘
           │                               │
           └───────────┬───────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │   CONCEPT DEFINITION         │
         │  • TILs (Tumor-Infiltrating  │
         │    Lymphocytes)              │
         │  • Necrosis                  │
         │  • Tumor Cellularity         │
         │  • Fibrosis/Therapy Effect   │
         │  • Mitotic Activity          │
         └─────────────┬───────────────┘
                       │
           ┌───────────┴────────────┐
           │                        │
           ▼                        ▼
┌──────────────────┐      ┌──────────────────┐
│   TCAV Module    │      │   MIL Classifier │
│  • CAV Learning  │      │  • Attention MIL │
│  • Concept Score │      │  • pCR Prediction│
│  • Attribution   │      │  • Heatmaps      │
└────────┬─────────┘      └────────┬─────────┘
         │                         │
         └────────────┬────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   INTERPRETABILITY     │
         │   • Attention Heatmaps │
         │   • Concept Scores     │
         │   • Prototype Matching │
         │   • Pathologist Review │
         └────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  OUTPUT: pCR Prediction│
         │  + Explanations        │
         └────────────────────────┘
```

---

## Environment Setup

### Hardware Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 3090/4090 or A100)
- **RAM**: 64GB+ recommended for WSI processing
- **Storage**: 500GB+ SSD for datasets and models
- **CPU**: 16+ cores for parallel tile processing

### Software Stack

#### Base Environment
```bash
# Create conda environment
conda create -n hovernet_tcav python=3.8
conda activate hovernet_tcav

# Install CUDA toolkit
conda install cudatoolkit=11.8 -c pytorch
```

#### Core Dependencies
```bash
# Deep learning frameworks
pip install torch==2.0.1 torchvision==0.15.2
pip install tensorflow==2.13.0

# Scientific computing
pip install numpy==1.24.3 scipy scikit-learn pandas

# Image processing
pip install openslide-python opencv-python Pillow
pip install albumentations staintools

# Visualization
pip install matplotlib seaborn plotly

# Pathology-specific
pip install histolab wholeslidedata
```

---

## Dataset Acquisition

### Primary Datasets with pCR Labels

#### 1. HER2-TUMOR-ROIS Dataset (TCIA)
**Source:** https://www.cancerimagingarchive.net/collection/her2-tumor-rois/

**Description:**
- 36 HER2+ breast cancer patients with pCR to trastuzumab +/- pertuzumab
- Pre-treatment H&E biopsy slides
- Complete pathologic response (pCR) labels

**Download Instructions:**
```bash
# Install NBIA Data Retriever
# Download from: https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images

# Steps:
# 1. Create TCIA account (free)
# 2. Navigate to HER2-TUMOR-ROIS collection
# 3. Add all cases to cart
# 4. Download manifest file (her2-tumor-rois-manifest.tcia)
# 5. Use NBIA Data Retriever to download

# Example using NBIA Data Retriever CLI
NBIADataRetriever --manifest her2-tumor-rois-manifest.tcia \
  --download-dir ./data/HER2-TUMOR-ROIS/
```

**Data Structure:**
```
HER2-TUMOR-ROIS/
├── patient_001/
│   ├── core_biopsy.svs
│   └── metadata.xml
├── patient_002/
│   └── ...
└── clinical_data.csv  # pCR labels, ER/PR/HER2 status
```

#### 2. Post-NAT-BRCA Dataset (TCIA)
**Source:** https://www.cancerimagingarchive.net/collection/post-nat-brca/

**Description:**
- 96 WSIs from 54 patients with residual disease after NAT
- Includes both pCR and non-pCR cases
- ER/PR/HER2 status available
- Tumor cellularity annotations

**Download Instructions:**
```bash
# Similar to HER2-TUMOR-ROIS
NBIADataRetriever --manifest post-nat-brca-manifest.tcia \
  --download-dir ./data/Post-NAT-BRCA/
```

#### 3. TCGA-BRCA (Optional - for additional training data)
**Source:** https://portal.gdc.cancer.gov/projects/TCGA-BRCA

**Description:**
- 1,097 breast cancer cases with H&E slides
- Limited treatment response data but rich molecular annotations
- Useful for pre-training feature extractors

**Download Instructions:**
```bash
# Install GDC Data Transfer Tool
wget https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip
unzip gdc-client_v1.6.1_Ubuntu_x64.zip

# Download TCGA-BRCA diagnostic slides
# 1. Go to https://portal.gdc.cancer.gov/
# 2. Filter: Project=TCGA-BRCA, Data Type=Slide Image
# 3. Add to cart and download manifest

./gdc-client download -m tcga-brca-manifest.txt \
  --dir ./data/TCGA-BRCA/
```

### Dataset Organization

```bash
# Create directory structure
mkdir -p data/{raw,processed,features,concepts}
mkdir -p data/raw/{HER2-TUMOR-ROIS,Post-NAT-BRCA,TCGA-BRCA}
mkdir -p models/{hovernet,tcav,mil}
mkdir -p results/{heatmaps,concepts,validation}
```

### Dataset Preprocessing Script
```python
# scripts/prepare_datasets.py
import pandas as pd
from pathlib import Path

def prepare_her2_dataset():
    """Prepare HER2-TUMOR-ROIS with pCR labels"""
    data_dir = Path("data/raw/HER2-TUMOR-ROIS")

    # Load clinical data
    clinical = pd.read_csv(data_dir / "clinical_data.csv")

    # Create dataset manifest
    manifest = []
    for patient_dir in data_dir.glob("patient_*"):
        patient_id = patient_dir.name
        slide_path = list(patient_dir.glob("*.svs"))[0]

        patient_data = clinical[clinical['patient_id'] == patient_id].iloc[0]

        manifest.append({
            'patient_id': patient_id,
            'slide_path': str(slide_path),
            'pCR': patient_data['pCR'],  # 1=complete response, 0=no pCR
            'HER2_status': patient_data['HER2'],
            'ER_status': patient_data['ER'],
            'PR_status': patient_data['PR']
        })

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv("data/processed/her2_manifest.csv", index=False)
    print(f"Prepared {len(manifest_df)} cases")
    print(f"pCR cases: {manifest_df['pCR'].sum()}")
    print(f"Non-pCR cases: {(~manifest_df['pCR']).sum()}")

    return manifest_df

if __name__ == "__main__":
    prepare_her2_dataset()
```

---

## Phase 1: HoverNet Implementation

### 1.1 Clone and Install HoverNet

```bash
cd /Users/rafik.salama/Codebase/HoverNet

# Clone official HoverNet repository
git clone https://github.com/vqdang/hover_net.git
cd hover_net

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
mkdir -p pretrained_models
cd pretrained_models

# Download HoverNet weights trained on PanNuke dataset
wget https://github.com/vqdang/hover_net/releases/download/v1.0/hovernet_fast_pannuke_type_tf2pytorch.tar
tar -xvf hovernet_fast_pannuke_type_tf2pytorch.tar

cd ../..
```

### 1.2 HoverNet Inference Pipeline

```python
# scripts/hovernet_inference.py
import numpy as np
import cv2
from pathlib import Path
from hover_net.run_infer import InferManager
import openslide

class HoverNetSegmentation:
    def __init__(self, model_path):
        """Initialize HoverNet model for nuclei segmentation"""
        self.model = InferManager(model_path=model_path)

    def segment_wsi_tiles(self, wsi_path, tile_size=512, overlap=128):
        """
        Segment nuclei in WSI tiles

        Args:
            wsi_path: Path to whole-slide image
            tile_size: Size of tiles for processing
            overlap: Overlap between tiles

        Returns:
            Dictionary with segmentation masks and classifications
        """
        slide = openslide.OpenSlide(wsi_path)

        # Get slide dimensions at desired magnification (20x)
        level = 0  # Highest resolution
        width, height = slide.level_dimensions[level]

        results = {
            'instance_maps': [],
            'type_maps': [],
            'nuclei_features': []
        }

        # Process tiles
        stride = tile_size - overlap
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # Extract tile
                tile = slide.read_region((x, y), level, (tile_size, tile_size))
                tile_rgb = np.array(tile.convert('RGB'))

                # Run HoverNet inference
                output = self.model.process_tile(tile_rgb)

                # Store results
                results['instance_maps'].append({
                    'position': (x, y),
                    'mask': output['inst_map']
                })
                results['type_maps'].append({
                    'position': (x, y),
                    'types': output['type_map']
                })

                # Extract nuclei features
                features = self.extract_nuclei_features(output)
                results['nuclei_features'].extend(features)

        slide.close()
        return results

    def extract_nuclei_features(self, hovernet_output):
        """
        Extract morphological features from segmented nuclei

        Features:
        - Nuclear area, perimeter, circularity
        - Nuclear type (tumor, lymphocyte, stromal, etc.)
        - Spatial density
        """
        inst_map = hovernet_output['inst_map']
        type_map = hovernet_output['type_map']

        nuclei_features = []

        # Get unique nucleus IDs
        nucleus_ids = np.unique(inst_map)[1:]  # Skip background (0)

        for nuc_id in nucleus_ids:
            mask = (inst_map == nuc_id).astype(np.uint8)

            # Compute morphological features
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue

            contour = contours[0]
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # Circularity
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0

            # Get nucleus type
            nucleus_type = type_map[mask > 0][0]  # 1=tumor, 2=lymphocyte, etc.

            # Centroid
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = 0, 0

            nuclei_features.append({
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'type': nucleus_type,
                'centroid': (cx, cy)
            })

        return nuclei_features

    def compute_slide_level_features(self, nuclei_features):
        """
        Aggregate nuclei features to slide-level metrics

        Returns:
        - TIL density (lymphocytes per mm²)
        - Tumor cell density
        - Necrosis markers
        - Mitotic index
        """
        features_array = pd.DataFrame(nuclei_features)

        # Count by type (assuming HoverNet type labels)
        # 1: Inflammatory, 2: Epithelial (tumor), 3: Stromal, 4: Dead
        type_counts = features_array['type'].value_counts()

        total_nuclei = len(features_array)

        metrics = {
            'lymphocyte_density': type_counts.get(1, 0) / total_nuclei,
            'tumor_cell_density': type_counts.get(2, 0) / total_nuclei,
            'stromal_density': type_counts.get(3, 0) / total_nuclei,
            'necrotic_density': type_counts.get(4, 0) / total_nuclei,
            'mean_nuclear_area': features_array['area'].mean(),
            'std_nuclear_area': features_array['area'].std(),
            'mean_circularity': features_array['circularity'].mean()
        }

        return metrics

# Usage example
if __name__ == "__main__":
    model_path = "pretrained_models/hovernet_fast_pannuke_type_tf2pytorch.tar"
    segmenter = HoverNetSegmentation(model_path)

    # Process a WSI
    wsi_path = "data/raw/HER2-TUMOR-ROIS/patient_001/core_biopsy.svs"
    results = segmenter.segment_wsi_tiles(wsi_path)

    # Compute slide-level features
    slide_features = segmenter.compute_slide_level_features(results['nuclei_features'])
    print("Slide-level features:", slide_features)
```

### 1.3 Stain Normalization Integration

```python
# scripts/stain_normalization.py
import staintools
import numpy as np

class MacenkoNormalizer:
    """Macenko stain normalization for H&E images"""

    def __init__(self, reference_image_path=None):
        self.normalizer = staintools.StainNormalizer(method='macenko')

        if reference_image_path:
            ref_image = staintools.read_image(reference_image_path)
            self.normalizer.fit(ref_image)
        else:
            # Use default reference from a well-stained image
            self.fit_default_reference()

    def fit_default_reference(self):
        """Fit normalizer with a standard reference"""
        # This would be a high-quality H&E image selected manually
        ref_image = staintools.read_image("data/reference_images/standard_he.png")
        self.normalizer.fit(ref_image)

    def normalize(self, image):
        """Normalize H&E stain of input image"""
        try:
            normalized = self.normalizer.transform(image)
            return normalized
        except Exception as e:
            print(f"Normalization failed: {e}")
            return image  # Return original if normalization fails

# Integration with HoverNet pipeline
def preprocess_tile_for_hovernet(tile_rgb, normalizer):
    """Preprocess tile with stain normalization before HoverNet"""
    normalized_tile = normalizer.normalize(tile_rgb)
    return normalized_tile
```

---

## Phase 2: TCAV Integration

### 2.1 Install TCAV Framework

```bash
# Clone official TCAV repository
git clone https://github.com/tensorflow/tcav.git
cd tcav
pip install -e .

cd /Users/rafik.salama/Codebase/HoverNet
```

### 2.2 Define Pathological Concepts

Based on the research file, key concepts for pCR prediction:

```python
# scripts/define_concepts.py
from pathlib import Path
import shutil

class ConceptDatasetBuilder:
    """Build concept datasets for TCAV analysis"""

    def __init__(self, base_dir="data/concepts"):
        self.base_dir = Path(base_dir)
        self.concepts = {
            'high_TILs': 'Dense tumor-infiltrating lymphocytes',
            'low_TILs': 'Sparse lymphocytic infiltration',
            'geographic_necrosis': 'Large areas of tumor necrosis',
            'viable_tumor': 'High density of viable tumor cells',
            'fibrosis': 'Therapy-induced fibrosis/scarring',
            'high_mitosis': 'High mitotic activity',
            'low_mitosis': 'Low mitotic figures',
            'poor_differentiation': 'Poorly differentiated tumor',
            'well_differentiation': 'Well-differentiated tumor'
        }

        # Create concept directories
        for concept_name in self.concepts.keys():
            (self.base_dir / concept_name).mkdir(parents=True, exist_ok=True)

    def collect_concept_examples(self, concept_name, source_wsi_paths,
                                  hovernet_results, n_examples=50):
        """
        Collect example patches for a concept using HoverNet features

        Args:
            concept_name: Name of concept (e.g., 'high_TILs')
            source_wsi_paths: List of WSI paths to extract from
            hovernet_results: HoverNet segmentation results
            n_examples: Number of example patches to collect
        """
        concept_dir = self.base_dir / concept_name

        # Define heuristics for each concept based on HoverNet features
        if concept_name == 'high_TILs':
            # Find tiles with high lymphocyte density
            selector = lambda features: features['lymphocyte_density'] > 0.3
        elif concept_name == 'geographic_necrosis':
            # Find tiles with high necrotic nucleus density
            selector = lambda features: features['necrotic_density'] > 0.4
        elif concept_name == 'viable_tumor':
            # High tumor cell density, low necrosis
            selector = lambda features: (features['tumor_cell_density'] > 0.5 and
                                         features['necrotic_density'] < 0.1)
        # ... define other concept selectors

        # Extract matching tiles
        collected = 0
        for wsi_path in source_wsi_paths:
            if collected >= n_examples:
                break

            # Get tiles matching concept criteria
            matching_tiles = self.extract_matching_tiles(
                wsi_path, hovernet_results[wsi_path], selector
            )

            # Save to concept directory
            for i, tile in enumerate(matching_tiles):
                if collected >= n_examples:
                    break
                save_path = concept_dir / f"{wsi_path.stem}_{i}.png"
                cv2.imwrite(str(save_path), tile)
                collected += 1

        print(f"Collected {collected} examples for concept '{concept_name}'")

    def create_random_counterexamples(self, n_examples=100):
        """Create random patches as negative examples for TCAV"""
        random_dir = self.base_dir / "random"
        random_dir.mkdir(exist_ok=True)

        # Extract random patches from WSIs
        # These serve as the "negative" set for TCAV
        # Implementation similar to collect_concept_examples but random sampling
        pass

# Manual annotation approach (higher quality)
def create_concept_annotation_tool():
    """
    Interactive tool for pathologists to annotate concept regions

    This would be a simple GUI where pathologists can:
    1. View WSI regions
    2. Draw bounding boxes around concept examples
    3. Save labeled regions
    """
    # Implementation using napari or similar visualization tool
    pass
```

### 2.3 TCAV Implementation

```python
# scripts/tcav_analysis.py
import tensorflow as tf
from tcav import tcav
from tcav import model_wrapper
from tcav import activation_generator
from tcav import cav
import numpy as np

class PathologyTCAV:
    """TCAV for pathology foundation models"""

    def __init__(self, model, layer_name, concept_dir="data/concepts"):
        """
        Initialize TCAV analyzer

        Args:
            model: Pre-trained foundation model (e.g., CONCH, TITAN)
            layer_name: Layer to analyze (typically penultimate layer)
            concept_dir: Directory containing concept image sets
        """
        self.model = model
        self.layer_name = layer_name
        self.concept_dir = Path(concept_dir)

        # Wrap model for TCAV
        self.model_wrapper = self._wrap_model()

    def _wrap_model(self):
        """Wrap model for TCAV interface"""
        class PathologyModelWrapper(model_wrapper.ModelWrapper):
            def __init__(self, model, layer_name):
                super().__init__()
                self.model = model
                self.layer_name = layer_name

            def get_gradient(self, acts, y, bottleneck_name):
                """Get gradients of prediction w.r.t. activations"""
                # Implementation depends on model architecture
                pass

            def get_predictions(self, imgs):
                """Get model predictions"""
                return self.model.predict(imgs)

        return PathologyModelWrapper(self.model, self.layer_name)

    def compute_tcav_scores(self, target_class='pCR', concepts=None):
        """
        Compute TCAV scores for each concept

        Args:
            target_class: Target class to explain (e.g., 'pCR')
            concepts: List of concept names to test

        Returns:
            Dictionary of concept names to TCAV scores
        """
        if concepts is None:
            concepts = [d.name for d in self.concept_dir.iterdir() if d.is_dir()]

        tcav_scores = {}

        for concept_name in concepts:
            print(f"Computing TCAV score for concept: {concept_name}")

            # Load concept images
            concept_imgs = self.load_concept_images(concept_name)
            random_imgs = self.load_concept_images("random")

            # Generate activations
            concept_acts = self.generate_activations(concept_imgs)
            random_acts = self.generate_activations(random_imgs)

            # Train CAV (linear classifier in activation space)
            concept_cav = cav.get_or_train_cav(
                concepts=[concept_name, "random"],
                bottleneck=self.layer_name,
                acts={concept_name: concept_acts, "random": random_acts},
                cav_dir="models/tcav/cavs"
            )

            # Compute TCAV score
            # This measures: % of target_class predictions that increase
            # when input is perturbed toward concept
            tcav_score = self.directional_derivative_score(
                concept_cav, target_class
            )

            tcav_scores[concept_name] = tcav_score

            print(f"  TCAV score: {tcav_score:.3f}")

        return tcav_scores

    def load_concept_images(self, concept_name):
        """Load all images for a concept"""
        concept_path = self.concept_dir / concept_name
        image_paths = list(concept_path.glob("*.png"))

        images = []
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

        return np.array(images)

    def generate_activations(self, images):
        """Generate activations at specified layer"""
        # Create intermediate model up to target layer
        intermediate_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(self.layer_name).output
        )

        activations = intermediate_model.predict(images, batch_size=32)
        return activations

    def directional_derivative_score(self, concept_cav, target_class):
        """
        Compute TCAV score via directional derivatives

        TCAV score = fraction of test examples where:
        ∂(prediction)/∂(concept direction) > 0
        """
        # Load test set (slides with pCR labels)
        test_images, test_labels = self.load_test_set()

        # Get target class examples
        target_indices = np.where(test_labels == target_class)[0]
        target_images = test_images[target_indices]

        # Compute gradients
        positive_count = 0
        for img in target_images:
            grad = self.compute_directional_derivative(img, concept_cav)
            if grad > 0:
                positive_count += 1

        tcav_score = positive_count / len(target_images)
        return tcav_score

    def compute_directional_derivative(self, image, concept_cav):
        """Compute gradient of prediction in concept direction"""
        with tf.GradientTape() as tape:
            # Forward pass
            activations = self.model(image[np.newaxis, ...])

            # Project gradient onto CAV direction
            grads = tape.gradient(activations, image)
            directional_grad = np.dot(grads.flatten(), concept_cav.cavs[0])

        return directional_grad

    def visualize_tcav_results(self, tcav_scores, save_path):
        """Create visualization of TCAV concept importance"""
        import matplotlib.pyplot as plt

        concepts = list(tcav_scores.keys())
        scores = list(tcav_scores.values())

        plt.figure(figsize=(12, 6))
        plt.barh(concepts, scores)
        plt.xlabel('TCAV Score (Concept Importance)', fontsize=12)
        plt.ylabel('Pathological Concepts', fontsize=12)
        plt.title('Concept Attribution for pCR Prediction', fontsize=14)
        plt.xlim([0, 1])
        plt.axvline(x=0.5, color='r', linestyle='--', label='Baseline')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved TCAV visualization to {save_path}")

# Usage
if __name__ == "__main__":
    # Load foundation model (e.g., CONCH)
    foundation_model = load_conch_model()  # Placeholder

    # Initialize TCAV
    tcav_analyzer = PathologyTCAV(
        model=foundation_model,
        layer_name='penultimate_layer',
        concept_dir="data/concepts"
    )

    # Compute scores
    scores = tcav_analyzer.compute_tcav_scores(target_class='pCR')

    # Visualize
    tcav_analyzer.visualize_tcav_results(
        scores,
        save_path="results/concepts/tcav_scores.png"
    )
```

---

## Phase 3: MIL Model & Interpretability

### 3.1 Attention-Based MIL Architecture

```python
# scripts/mil_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning for WSI classification
    Based on CLAM architecture
    """

    def __init__(self, feature_dim=1024, hidden_dim=256, n_classes=2, dropout=0.25):
        super(AttentionMIL, self).__init__()

        self.feature_dim = feature_dim
        self.n_classes = n_classes

        # Attention network
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, features):
        """
        Args:
            features: [N, feature_dim] - N patch features from one WSI

        Returns:
            logits: [n_classes]
            attention_weights: [N] - importance of each patch
        """
        # Compute attention scores
        attention_logits = self.attention_net(features)  # [N, 1]
        attention_weights = F.softmax(attention_logits, dim=0)  # [N, 1]

        # Weighted aggregation
        slide_features = torch.sum(attention_weights * features, dim=0)  # [feature_dim]

        # Classification
        logits = self.classifier(slide_features)  # [n_classes]

        return logits, attention_weights.squeeze()

    def get_attention_heatmap(self, features, patch_coords):
        """
        Generate attention heatmap for visualization

        Args:
            features: Patch features
            patch_coords: [(x, y), ...] coordinates of patches

        Returns:
            Heatmap array
        """
        _, attention_weights = self.forward(features)

        # Map attention to spatial coordinates
        heatmap = create_spatial_heatmap(
            attention_weights.cpu().detach().numpy(),
            patch_coords
        )

        return heatmap

def create_spatial_heatmap(attention_weights, patch_coords, slide_dims):
    """Create 2D heatmap from patch attention scores"""
    import numpy as np

    heatmap = np.zeros(slide_dims)

    for weight, (x, y) in zip(attention_weights, patch_coords):
        heatmap[y:y+256, x:x+256] = weight

    return heatmap
```

### 3.2 Training Pipeline

```python
# scripts/train_mil.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class WSIDataset(Dataset):
    """Dataset for WSI features with pCR labels"""

    def __init__(self, manifest_path, features_dir):
        self.manifest = pd.read_csv(manifest_path)
        self.features_dir = Path(features_dir)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        # Load precomputed features for this WSI
        features_path = self.features_dir / f"{row['patient_id']}_features.pt"
        features = torch.load(features_path)

        label = int(row['pCR'])  # 0 or 1

        return features, label

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits, _ = model(features)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits, _ = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            probs = F.softmax(logits, dim=1)[:, 1]  # pCR probability
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Compute AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, auc

# Main training loop
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    train_dataset = WSIDataset(
        manifest_path="data/processed/train_manifest.csv",
        features_dir="data/features/train"
    )
    val_dataset = WSIDataset(
        manifest_path="data/processed/val_manifest.csv",
        features_dir="data/features/val"
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize model
    model = AttentionMIL(
        feature_dim=1024,  # Depends on foundation model
        hidden_dim=256,
        n_classes=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Training
    best_auc = 0
    for epoch in range(50):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/50")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "models/mil/best_model.pth")
            print(f"  Saved new best model (AUC: {val_auc:.4f})")

    print(f"\nBest validation AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()
```

---

## Validation & Testing

### Faithfulness Tests

```python
# scripts/validation/faithfulness_tests.py

def ablation_study(model, wsi_features, patch_coords, top_k=10):
    """
    Test if removing top-attention regions reduces prediction confidence

    Args:
        model: Trained MIL model
        wsi_features: All patch features [N, D]
        patch_coords: Coordinates of patches
        top_k: Number of top patches to ablate

    Returns:
        Performance drop metric
    """
    # Original prediction
    original_logits, attention_weights = model(wsi_features)
    original_prob = F.softmax(original_logits, dim=1)[1].item()

    # Get top-k attended patches
    top_indices = torch.topk(attention_weights, k=top_k).indices

    # Remove top patches
    mask = torch.ones(len(wsi_features), dtype=torch.bool)
    mask[top_indices] = False
    ablated_features = wsi_features[mask]

    # Prediction after ablation
    ablated_logits, _ = model(ablated_features)
    ablated_prob = F.softmax(ablated_logits, dim=1)[1].item()

    # Compute drop
    confidence_drop = original_prob - ablated_prob

    print(f"Original pCR probability: {original_prob:.3f}")
    print(f"After removing top-{top_k} patches: {ablated_prob:.3f}")
    print(f"Confidence drop: {confidence_drop:.3f}")

    return confidence_drop
```

### Clinical Plausibility

```python
# scripts/validation/pathologist_review.py

def generate_explanation_report(model, wsi_path, tcav_scores):
    """
    Generate human-readable explanation for pathologist review

    Returns:
        Structured report with:
        - Attention heatmap
        - Top concepts
        - Representative patches
    """
    report = {
        'wsi_id': wsi_path.stem,
        'prediction': {
            'pCR_probability': 0.85,
            'predicted_class': 'pCR'
        },
        'attention_regions': [
            {
                'rank': 1,
                'location': (1200, 3400),
                'attention_weight': 0.23,
                'thumbnail': 'patch_001.png'
            },
            # ... more regions
        ],
        'concept_attribution': [
            {
                'concept': 'high_TILs',
                'tcav_score': 0.87,
                'interpretation': 'Model strongly relies on lymphocytic infiltration'
            },
            {
                'concept': 'geographic_necrosis',
                'tcav_score': 0.72,
                'interpretation': 'Necrosis presence increases pCR prediction'
            },
            {
                'concept': 'viable_tumor',
                'tcav_score': -0.65,
                'interpretation': 'High tumor cellularity decreases pCR prediction'
            }
        ],
        'hovernet_features': {
            'TIL_density': 0.42,
            'tumor_cell_density': 0.18,
            'necrotic_density': 0.31,
            'mean_nuclear_area': 87.5
        }
    }

    return report
```

---

## Timeline & Milestones

### Phase 1: Setup & Data (Weeks 1-2)
- ✓ Environment setup
- ✓ Dataset download (HER2-TUMOR-ROIS, Post-NAT-BRCA)
- ✓ HoverNet installation
- ✓ TCAV installation

**Deliverable:** Datasets organized, software installed

### Phase 2: Feature Extraction (Weeks 3-4)
- Run HoverNet on all WSIs
- Extract nuclei features
- Implement stain normalization
- Compute slide-level metrics

**Deliverable:** Feature database with morphological metrics

### Phase 3: Concept Definition (Week 5)
- Define pathological concepts
- Collect concept example patches
- Validate with pathologist
- Build concept datasets

**Deliverable:** Concept library with 50+ examples per concept

### Phase 4: TCAV Implementation (Week 6)
- Train CAVs for each concept
- Compute TCAV scores
- Validate concept importance
- Generate concept visualizations

**Deliverable:** TCAV scores showing which concepts drive pCR predictions

### Phase 5: MIL Model Training (Weeks 7-8)
- Extract foundation model features
- Train attention-based MIL
- Optimize hyperparameters
- Validate on hold-out set

**Deliverable:** Trained MIL model with AUC > 0.80

### Phase 6: Interpretability Integration (Week 9)
- Generate attention heatmaps
- Link heatmaps to concepts
- Implement faithfulness tests
- Create explanation reports

**Deliverable:** Integrated interpretability system

### Phase 7: Validation (Week 10)
- Ablation studies
- Stability tests (stain variations)
- Pathologist review
- Final documentation

**Deliverable:** Validation report, pathologist feedback

---

## Expected Results

### Quantitative Metrics
- **Model Performance:** AUC 0.80-0.85 for pCR prediction
- **Concept Attribution:** TCAV scores > 0.70 for key concepts (TILs, necrosis)
- **Faithfulness:** >30% confidence drop when removing top 10% attended regions
- **Stability:** >0.75 IoU between attention maps across stain variations

### Qualitative Outcomes
- Pathologist-validated explanations
- Human-readable concept summaries
- Visual attention heatmaps aligned with histology
- Case-based examples of model reasoning

### Deliverables
1. Trained HoverNet segmentation pipeline
2. TCAV concept attribution framework
3. Interpretable MIL pCR prediction model
4. Validation report with pathologist review
5. Open-source codebase on GitHub

---

## References

1. **HoverNet:** Graham et al., "Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images," Medical Image Analysis, 2019.
2. **TCAV:** Kim et al., "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors," ICML 2018.
3. **CLAM:** Lu et al., "Data-efficient and weakly supervised computational pathology on whole-slide images," Nature Biomedical Engineering, 2021.
4. **Stain Normalization:** Macenko et al., "A method for normalizing histology slides for quantitative analysis," ISBI 2009.
5. **HistoQC:** Janowczyk et al., "HistoQC: An Open-Source Quality Control Tool for Digital Pathology Slides," JCO Clinical Cancer Informatics, 2019.

---

## Next Steps

1. **Review this plan** and confirm approach
2. **Set up development environment** (GPU access, software installation)
3. **Download HER2-TUMOR-ROIS dataset** (primary dataset with pCR labels)
4. **Run HoverNet on sample slides** to validate pipeline
5. **Schedule pathologist consultation** for concept definition
6. **Begin Phase 1 implementation**

---

**Document Status:** Draft v1.0
**Last Updated:** 2025-11-19
**Author:** AI Research Team
**Contact:** rafik.salama@codebase