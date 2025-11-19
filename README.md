# HoverNet + TCAV: Interpretable pCR Prediction from H&E Slides

[![Tests](https://github.com/yourusername/hovernet-tcav-pcr/workflows/Tests/badge.svg)](https://github.com/yourusername/hovernet-tcav-pcr/actions)
[![codecov](https://codecov.io/gh/yourusername/hovernet-tcav-pcr/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/hovernet-tcav-pcr)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Interpretable AI system for predicting pathological complete response (pCR) to neoadjuvant chemotherapy from H&E whole-slide images using HoverNet nuclei segmentation and TCAV concept-based explanations.**

---

## 🎯 Overview

This project implements a **Test-Driven Development (TDD)** approach to build an interpretable deep learning system that:

- **Predicts pCR** from pre-treatment H&E biopsy slides (AUC > 0.80)
- **Explains predictions** using pathological concepts (TILs, necrosis, tumor cellularity)
- **Provides visual attention heatmaps** highlighting important regions
- **Ensures clinical validity** through rigorous validation

### Key Components

1. **HoverNet**: Nuclei segmentation and classification
2. **TCAV**: Testing with Concept Activation Vectors for interpretability
3. **Attention MIL**: Multiple Instance Learning for WSI-level prediction
4. **Validation Framework**: Faithfulness, stability, and clinical plausibility tests

---

## 📊 Success Criteria

### Phase 1: HoverNet Segmentation ✅
- [ ] Dice coefficient > 0.75
- [ ] TIL density correlation > 0.7
- [ ] Processing speed < 2s per tile
- [ ] 80%+ test coverage

### Phase 2: TCAV Integration ✅
- [ ] 8+ pathological concepts defined
- [ ] TCAV scores clinically plausible
- [ ] Stability across seeds (std < 0.1)
- [ ] 75%+ test coverage

### Phase 3: MIL Model ✅
- [ ] Validation AUC > 0.80
- [ ] Balanced accuracy > 0.75
- [ ] Attention on tissue (>95%)
- [ ] 80%+ test coverage

### Phase 4: Interpretability ✅
- [ ] Ablation drop > 20%
- [ ] Pathologist agreement > 70%
- [ ] >90% explanation coverage

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (16GB+ VRAM recommended)
- 64GB+ RAM
- 500GB+ storage

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/hovernet-tcav-pcr.git
cd hovernet-tcav-pcr

# Create conda environment
conda create -n hovernet_tcav python=3.8 -y
conda activate hovernet_tcav

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .

# Verify installation
pytest tests/ -m "unit and not gpu and not data" -v
```

### Download Data

See [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) for detailed instructions on downloading:
- HER2-TUMOR-ROIS dataset (36 pCR cases)
- Post-NAT-BRCA dataset (96 WSIs)

---

## 📁 Project Structure

```
hovernet-tcav-pcr/
├── src/                          # Source code
│   ├── hovernet_pipeline/        # HoverNet segmentation
│   │   ├── segmentation.py
│   │   ├── features.py
│   │   └── metrics.py
│   ├── tcav_integration/         # TCAV concept attribution
│   │   ├── concept_builder.py
│   │   ├── tcav_core.py
│   │   └── cli.py
│   ├── mil_model/                # MIL classifier
│   │   ├── attention_mil.py
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── metrics.py
│   ├── data_processing/          # Data preprocessing
│   │   ├── stain_normalization.py
│   │   └── quality_control.py
│   ├── interpretability/         # Interpretability tools
│   │   ├── faithfulness.py
│   │   └── visualization.py
│   └── utils/                    # Utilities
│
├── tests/                        # Test suite (TDD)
│   ├── unit/                     # Unit tests
│   │   ├── test_hovernet_segmentation.py
│   │   ├── test_tcav_integration.py
│   │   └── test_mil_model.py
│   ├── integration/              # Integration tests
│   ├── validation/               # Success criteria tests
│   │   └── test_success_criteria.py
│   ├── fixtures/                 # Test data
│   └── conftest.py               # Pytest configuration
│
├── data/                         # Data directory
│   ├── raw/                      # Raw WSIs
│   ├── processed/                # Processed data
│   ├── features/                 # Extracted features
│   ├── concepts/                 # Concept images
│   └── models/                   # Saved models
│
├── scripts/                      # Utility scripts
├── notebooks/                    # Jupyter notebooks
├── docs/                         # Documentation
│   └── TESTING_STRATEGY.md       # TDD strategy
├── config/                       # Configuration files
│
├── PROTOTYPE_IMPLEMENTATION_PLAN.md
├── QUICK_START_GUIDE.md
├── TESTING_STRATEGY.md
│
├── requirements.txt              # Core dependencies
├── requirements-dev.txt          # Development dependencies
├── setup.py                      # Package setup
├── pytest.ini                    # Pytest configuration
├── pyproject.toml                # Project metadata
├── Makefile                      # Common tasks
└── README.md                     # This file
```

---

## 🧪 Test-Driven Development

This project follows **strict TDD** principles:

### Running Tests

```bash
# Run all unit tests
make test-unit

# Run integration tests
make test-integration

# Run validation tests
make test-validation

# Run all tests with coverage
make test-all

# Run specific phase tests
pytest -m phase1
pytest -m phase2
pytest -m phase3

# Run success criteria tests
pytest -m success_criteria

# Generate coverage report
make coverage
```

### Test Categories

- **Unit Tests** (80% of tests): Test individual functions in isolation
- **Integration Tests** (15%): Test component interactions
- **Validation Tests** (5%): Validate phase success criteria

### Success Criteria Validation

Before proceeding to the next phase, run:

```bash
pytest tests/validation/test_success_criteria.py -v
```

All success criteria must pass! ✅

---

## 📖 Usage Examples

### 1. Segment Nuclei with HoverNet

```python
from hovernet_pipeline.segmentation import segment_nuclei
import cv2

# Load H&E tile
tile = cv2.imread("data/sample_tile.png")

# Segment nuclei
result = segment_nuclei(tile)

# Extract features
from hovernet_pipeline.features import extract_nuclei_features
features = extract_nuclei_features(result)

print(f"Detected {len(features)} nuclei")
```

### 2. Compute TCAV Concept Importance

```python
from tcav_integration.tcav_core import compute_tcav_scores_for_concepts

# Compute TCAV scores
tcav_scores = compute_tcav_scores_for_concepts(
    model=foundation_model,
    concepts=['high_TILs', 'necrosis', 'viable_tumor'],
    target_class='pCR'
)

print("Concept importance:")
for concept, score in tcav_scores.items():
    print(f"  {concept}: {score:.3f}")
```

### 3. Train MIL Model

```python
from mil_model.train import train_model

# Train attention-based MIL
model, metrics = train_model(
    train_manifest="data/processed/train_manifest.csv",
    val_manifest="data/processed/val_manifest.csv",
    features_dir="data/features/",
    n_epochs=50,
    early_stopping_patience=10
)

print(f"Validation AUC: {metrics['val_auc']:.3f}")
```

### 4. Generate Interpretable Predictions

```python
from interpretability.explain import generate_explanation_report

# Get prediction with explanation
report = generate_explanation_report(
    model=mil_model,
    wsi_path="data/raw/patient_001.svs",
    tcav_analyzer=tcav_analyzer
)

print(f"Prediction: {report['prediction']['pCR_probability']:.2f}")
print(f"Top concepts: {report['concept_attribution'][:3]}")
```

---

## 📊 Datasets

### Primary Dataset: HER2-TUMOR-ROIS

- **Source**: The Cancer Imaging Archive (TCIA)
- **Size**: 36 HER2+ breast cancer patients
- **Labels**: Complete pCR response labels
- **Download**: https://www.cancerimagingarchive.net/collection/her2-tumor-rois/

### Secondary Dataset: Post-NAT-BRCA

- **Source**: TCIA
- **Size**: 96 WSIs from 54 patients
- **Labels**: pCR and non-pCR cases
- **Download**: https://www.cancerimagingarchive.net/collection/post-nat-brca/

---

## 🔬 Development Workflow

### 1. Write Tests First (RED)

```bash
# Create test file
touch tests/unit/test_new_feature.py

# Write failing test
# Run tests - should FAIL
pytest tests/unit/test_new_feature.py
```

### 2. Implement Feature (GREEN)

```python
# Implement minimal code to pass test
# Run tests - should PASS
pytest tests/unit/test_new_feature.py
```

### 3. Refactor (REFACTOR)

```python
# Improve code while keeping tests passing
# Run all tests
make test-all
```

### 4. Check Coverage

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

---

## 🤝 Contributing

We follow strict TDD and code quality standards:

1. **Write tests first** - No code without tests
2. **Pass all tests** - 100% pass rate required
3. **Maintain coverage** - Minimum 80% coverage
4. **Format code** - `make format` before commit
5. **Lint code** - `make lint` must pass

### Pull Request Checklist

- [ ] All tests pass (`make test-all`)
- [ ] Coverage ≥ 80% (`make coverage`)
- [ ] Code formatted (`make format`)
- [ ] Linting passes (`make lint`)
- [ ] Documentation updated
- [ ] Success criteria validated

---

## 📚 Documentation

- [PROTOTYPE_IMPLEMENTATION_PLAN.md](PROTOTYPE_IMPLEMENTATION_PLAN.md) - Detailed technical plan
- [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) - Step-by-step setup
- [docs/TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md) - TDD strategy and success criteria

---

## 🏆 Phase Completion Status

| Phase | Status | Tests | Coverage | Success Criteria |
|-------|--------|-------|----------|------------------|
| Phase 1: HoverNet | 🔴 Not Started | 0/30 | 0% | 0/7 |
| Phase 2: TCAV | 🔴 Not Started | 0/25 | 0% | 0/7 |
| Phase 3: MIL Model | 🔴 Not Started | 0/35 | 0% | 0/7 |
| Phase 4: Interpretability | 🔴 Not Started | 0/20 | 0% | 0/7 |

**Overall Progress: 0/110 tests | 0/28 success criteria**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **HoverNet**: Graham et al., "Hover-Net: Simultaneous segmentation and classification of nuclei"
- **TCAV**: Kim et al., "Interpretability Beyond Feature Attribution"
- **CLAM**: Lu et al., "Data-efficient and weakly supervised computational pathology"
- **TCIA**: The Cancer Imaging Archive for public datasets

---

## 📧 Contact

- **Author**: AI Research Team
- **Email**: rafik.salama@codebase
- **GitHub**: https://github.com/yourusername/hovernet-tcav-pcr

---

## 🚀 Next Steps

1. **Review documentation**
   - Read [PROTOTYPE_IMPLEMENTATION_PLAN.md](PROTOTYPE_IMPLEMENTATION_PLAN.md)
   - Study [TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md)

2. **Set up environment**
   - Follow [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)
   - Run `make install` and `make test`

3. **Download datasets**
   - Get HER2-TUMOR-ROIS from TCIA
   - Set up data directory structure

4. **Start Phase 1**
   - Implement HoverNet segmentation
   - Pass all Phase 1 tests
   - Validate success criteria

**Let's build interpretable AI for pathology! 🔬**
