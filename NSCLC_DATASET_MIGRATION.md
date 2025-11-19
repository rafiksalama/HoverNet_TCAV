# Migration to NSCLC Datasets - HoverNet + TCAV Project

**Date:** 2025-11-19
**Change:** Migrating from breast cancer (pCR prediction) to NSCLC (Non-Small Cell Lung Cancer) subtype classification

---

## New Objective

**Previous:** Predict pathological complete response (pCR) to neoadjuvant chemotherapy in HER2+ breast cancer

**New:** Classify NSCLC subtypes (Adenocarcinoma vs. Squamous Cell Carcinoma) from H&E whole-slide images using interpretable AI

---

## NSCLC Datasets

### 1. **TCGA-LUAD** (Lung Adenocarcinoma) - PRIMARY DATASET
**Source:** GDC Data Portal (https://portal.gdc.cancer.gov/projects/TCGA-LUAD)

**Specifications:**
- **Cases:** 541 diagnostic WSIs from 478 patients
- **Format:** .svs files (FFPE H&E slides)
- **Labels:** Adenocarcinoma subtype
- **Resolution:** 20x, 40x magnification
- **Size:** ~800GB total
- **Access:** Public, free download via GDC Data Transfer Tool

**Download Instructions:**
```bash
# 1. Install GDC Data Transfer Tool
wget https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip
unzip gdc-client_v1.6.1_Ubuntu_x64.zip

# 2. Get manifest from GDC portal
# Navigate to: https://portal.gdc.cancer.gov/projects/TCGA-LUAD
# Click "Diagnostic Slide" → Download Manifest

# 3. Download slides
./gdc-client download -m gdc_manifest_LUAD.txt --dir data/TCGA-LUAD/
```

---

### 2. **TCGA-LUSC** (Lung Squamous Cell Carcinoma) - PRIMARY DATASET
**Source:** GDC Data Portal (https://portal.gdc.cancer.gov/projects/TCGA-LUSC)

**Specifications:**
- **Cases:** 512 diagnostic WSIs from 478 patients
- **Format:** .svs files (FFPE H&E slides)
- **Labels:** Squamous cell carcinoma subtype
- **Resolution:** 20x, 40x magnification
- **Size:** ~750GB total
- **Access:** Public, free download via GDC Data Transfer Tool

**Download Instructions:**
```bash
# Get manifest from GDC portal
# Navigate to: https://portal.gdc.cancer.gov/projects/TCGA-LUSC
# Click "Diagnostic Slide" → Download Manifest

./gdc-client download -m gdc_manifest_LUSC.txt --dir data/TCGA-LUSC/
```

---

### 3. **CPTAC-LSCC** (Clinical Proteomic Tumor Analysis Consortium) - VALIDATION DATASET
**Source:** The Cancer Imaging Archive (https://www.cancerimagingarchive.net/collection/cptac-lscc/)

**Specifications:**
- **Cases:** 367 WSIs from 51 patients
- **Format:** .svs files
- **Labels:** Lung squamous cell carcinoma + proteomic data
- **Additional Data:** Clinical, genomic, proteomic annotations
- **Access:** Public via TCIA

**Download Instructions:**
```bash
# Install NBIA Data Retriever
# Download from: https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images

# Get manifest from TCIA
# Navigate to: https://www.cancerimagingarchive.net/collection/cptac-lscc/
# Click "Download" → Select "Pathology Images"

NBIADataRetriever --manifest cptac-lscc-manifest.tcia \
  --download-dir ./data/CPTAC-LSCC/
```

---

### 4. **LungHist700** (2024) - SUPPLEMENTARY DATASET
**Source:** Scientific Data (https://www.nature.com/articles/s41597-024-03944-3)

**Specifications:**
- **Cases:** 691 high-resolution images (1200×1600 pixels)
- **Patients:** 45 patients
- **Types:** Adenocarcinoma, Squamous cell carcinoma, Normal tissue
- **Differentiation Levels:** 3 levels for each cancer type
- **Magnification:** 20x and 40x
- **Format:** High-resolution image patches
- **Published:** October 2024
- **Use Case:** Patch-level training, quality control

---

## Updated Research Goals

### Classification Task
**Binary Classification:**
- **Class 0:** Lung Adenocarcinoma (LUAD)
- **Class 1:** Lung Squamous Cell Carcinoma (LUSC)

**Why This Task:**
- Clinical relevance: Different subtypes require different treatments
- Diagnostic challenge: Accurate subtyping is critical for therapy selection
- Interpretability value: Pathologists need to understand model decisions
- Published benchmarks: Multiple papers report 85-95% accuracy on this task

---

## Key Pathological Concepts for NSCLC

### TCAV Concepts (Updated from Breast)

**Previous Breast Cancer Concepts:**
- high_TILs, low_TILs
- geographic_necrosis
- viable_tumor, fibrosis
- high_mitosis, low_mitosis

**New NSCLC Concepts:**

1. **Glandular Structures** (LUAD marker)
   - Acinar patterns
   - Papillary patterns
   - Lepidic growth

2. **Squamous Differentiation** (LUSC marker)
   - Keratinization
   - Intercellular bridges
   - Individual cell keratinization

3. **Tumor-Infiltrating Lymphocytes (TILs)**
   - High TIL density
   - Low TIL density

4. **Necrosis**
   - Geographic necrosis
   - Single cell necrosis

5. **Nuclear Features**
   - Nuclear pleomorphism
   - Prominent nucleoli
   - Hyperchromasia

6. **Architectural Patterns**
   - Solid growth pattern
   - Cribriform pattern
   - Invasive margin

7. **Stroma**
   - Desmoplastic stroma
   - Inflammatory stroma

---

## Updated HoverNet Features for NSCLC

### Nucleus Types (4 classes remain, but interpretation changes)

**Previous (Breast):**
1. Inflammatory/Lymphocyte
2. Epithelial/Tumor
3. Stromal
4. Necrotic

**New (NSCLC):**
1. **Lymphocyte/Inflammatory** - TILs, immune cells
2. **Tumor (Adenocarcinoma or Squamous)** - Malignant epithelial cells
3. **Stromal** - Fibroblasts, connective tissue
4. **Necrotic** - Dead/dying cells

### Morphological Features (unchanged)
- Area, perimeter, circularity
- Eccentricity, solidity, compactness
- Ellipse fit parameters
- Convex hull metrics

**Total: 15+ features per nucleus**

---

## Updated Success Metrics

### Phase 1: HoverNet Segmentation (NSCLC-specific)
- [x] Segment ≥80% of visible nuclei in NSCLC tissue
- [x] Extract 15+ morphological features
- [ ] TIL density correlation with manual annotation > 0.7

### Phase 2: TCAV Integration (NSCLC-specific)
- [ ] Define 7+ NSCLC pathological concepts
- [ ] Glandular structures → positive association with LUAD (score > 0.6)
- [ ] Squamous differentiation → positive association with LUSC (score > 0.6)
- [ ] TCAV scores clinically plausible (validated by pathologist)

### Phase 3: MIL Model Training (NSCLC-specific)
- [ ] Validation AUC > 0.85 for LUAD vs LUSC classification
- [ ] Balanced accuracy > 0.80
- [ ] Attention focuses on diagnostically relevant regions
- [ ] External validation on CPTAC-LSCC > 0.80 AUC

### Phase 4: Interpretability Validation (NSCLC-specific)
- [ ] Top concepts align with known NSCLC biomarkers (glandular, squamous)
- [ ] Pathologist agreement with explanations > 70%
- [ ] Ablation of glandular regions drops LUAD confidence > 20%

---

## Data Organization

```
data/
├── TCGA-LUAD/           # 541 WSIs, ~800GB
│   ├── raw/             # Downloaded .svs files
│   ├── tiles/           # Extracted 512×512 tiles
│   └── features/        # HoverNet features per tile
│
├── TCGA-LUSC/           # 512 WSIs, ~750GB
│   ├── raw/
│   ├── tiles/
│   └── features/
│
├── CPTAC-LSCC/          # 367 WSIs (validation)
│   ├── raw/
│   ├── tiles/
│   └── features/
│
├── LungHist700/         # 691 patches (supplementary)
│   └── patches/
│
└── concepts/            # TCAV concept examples
    ├── glandular_structures/
    ├── squamous_differentiation/
    ├── high_TILs/
    ├── necrosis/
    ├── nuclear_pleomorphism/
    ├── solid_pattern/
    └── random/          # Random counterexamples
```

---

## Clinical Relevance

### Why LUAD vs LUSC Classification Matters

**Treatment Differences:**
- **LUAD:** Often responds to targeted therapies (EGFR, ALK inhibitors)
- **LUSC:** Different mutation profile, different treatment protocols

**Diagnostic Challenge:**
- 10-15% of cases are difficult to classify
- Biopsy samples may be limited
- AI can assist in ambiguous cases

**Interpretability Importance:**
- Pathologists need to understand *why* the model classified as LUAD or LUSC
- TCAV concepts (glandular structures, keratinization) map to diagnostic criteria
- Model explanations must align with pathology training

---

## Migration Checklist

- [x] Research NSCLC datasets (TCGA-LUAD, TCGA-LUSC, CPTAC-LSCC)
- [ ] Update PROTOTYPE_IMPLEMENTATION_PLAN.md with NSCLC focus
- [ ] Update README.md with new objectives
- [ ] Modify concept definitions for NSCLC pathology
- [ ] Update synthetic H&E generator for lung tissue morphology
- [ ] Download TCGA-LUAD manifest (sample of 50 cases for prototyping)
- [ ] Download TCGA-LUSC manifest (sample of 50 cases for prototyping)
- [ ] Update test cases to use NSCLC terminology
- [ ] Validate with pathologist (if available)

---

## References

1. **TCGA-LUAD:** https://portal.gdc.cancer.gov/projects/TCGA-LUAD
2. **TCGA-LUSC:** https://portal.gdc.cancer.gov/projects/TCGA-LUSC
3. **CPTAC-LSCC:** https://www.cancerimagingarchive.net/collection/cptac-lscc/
4. **LungHist700:** https://www.nature.com/articles/s41597-024-03944-3
5. **Download Guide:** https://andrewjanowczyk.com/download-tcga-digital-pathology-images-ffpe/
6. **GDC API:** https://research.adfoucart.be/tcga-retrieval-gdc-api

---

## Timeline

**Immediate (Week 1):**
- Update all documentation
- Download 50 LUAD + 50 LUSC samples for prototyping
- Update visualization to show lung tissue

**Phase 2 (Weeks 2-4):**
- Define NSCLC concepts
- Collect concept examples from TCGA slides
- Train TCAV models

**Phase 3 (Weeks 5-8):**
- Train MIL classifier on LUAD vs LUSC
- Validate on CPTAC-LSCC

**Phase 4 (Weeks 9-10):**
- Interpretability validation
- Pathologist review (if available)

---

**Status:** READY TO PROCEED WITH NSCLC DATASETS ✅
