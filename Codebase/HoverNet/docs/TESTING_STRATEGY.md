# Testing Strategy & Success Criteria

## Overview

This document defines the **Test-Driven Development (TDD)** strategy for the HoverNet + TCAV prototype. Each phase has clearly defined success criteria that must be validated through automated tests before proceeding to the next phase.

---

## Testing Philosophy

### Test-Driven Development Workflow
```
1. Write test defining expected behavior (RED)
   ↓
2. Write minimal code to pass test (GREEN)
   ↓
3. Refactor while keeping tests passing (REFACTOR)
   ↓
4. Repeat
```

### Test Pyramid Structure
```
        /\
       /  \        E2E Tests (5%)
      /____\       - Full pipeline validation
     /      \      - Clinical scenarios
    /________\     Integration Tests (15%)
   /          \    - Multi-component workflows
  /____________\   - Data pipeline integration
 /              \  Unit Tests (80%)
/________________\ - Individual functions
                   - Edge cases & error handling
```

---

## Test Categories

### 1. Unit Tests (`tests/unit/`)
**Purpose:** Test individual functions and classes in isolation

**Coverage Requirements:** >80% line coverage, >70% branch coverage

**Characteristics:**
- Fast (<100ms per test)
- No external dependencies
- Use mocks for I/O operations
- Test one thing at a time

### 2. Integration Tests (`tests/integration/`)
**Purpose:** Test interactions between components

**Coverage Requirements:** All major workflows tested

**Characteristics:**
- May use real file I/O
- Test data pipelines
- Verify component interfaces
- Medium speed (<5s per test)

### 3. Validation Tests (`tests/validation/`)
**Purpose:** Validate model performance and success criteria

**Coverage Requirements:** All phase success criteria

**Characteristics:**
- Use real or realistic test data
- Measure performance metrics
- Compare against baselines
- May be slow (marked with `@pytest.mark.slow`)

---

## Phase Success Criteria

### Phase 1: HoverNet Segmentation ✅

#### Success Criteria
1. **Segmentation Quality**
   - [ ] Dice coefficient > 0.75 on validation set
   - [ ] Can segment nuclei in 95%+ of tissue regions
   - [ ] Classification accuracy > 70% (tumor vs. lymphocyte vs. stromal)

2. **Feature Extraction**
   - [ ] Extract 15+ morphological features per nucleus
   - [ ] Slide-level aggregation matches manual counts (±15%)
   - [ ] TIL density correlation with manual annotation > 0.7

3. **Performance**
   - [ ] Process 512x512 tile in <2 seconds (GPU)
   - [ ] Handle WSI of 100k+ tiles without memory errors
   - [ ] Robust to stain variations (normalized vs. non-normalized)

4. **Code Quality**
   - [ ] 80%+ test coverage
   - [ ] All tests pass
   - [ ] No critical bugs

#### Key Tests
```python
# tests/unit/test_hovernet_segmentation.py
def test_segment_single_tile()
def test_nuclei_classification()
def test_feature_extraction()
def test_stain_normalization_improves_consistency()

# tests/integration/test_hovernet_pipeline.py
def test_full_wsi_processing()
def test_slide_level_aggregation()

# tests/validation/test_phase1_success.py
def test_segmentation_dice_score()
def test_til_density_correlation()
def test_processing_speed_benchmark()
```

---

### Phase 2: TCAV Integration ✅

#### Success Criteria
1. **Concept Definition**
   - [ ] Define 8+ pathological concepts
   - [ ] Collect 50+ examples per concept
   - [ ] Inter-rater agreement > 0.7 (if manually annotated)

2. **TCAV Scores**
   - [ ] Compute TCAV scores for all concepts
   - [ ] Significant concepts (p < 0.05) align with clinical knowledge
   - [ ] High TILs concept → positive association with pCR (score > 0.6)
   - [ ] Viable tumor concept → negative association with pCR (score < 0.4)

3. **Stability**
   - [ ] TCAV scores consistent across random seeds (std < 0.1)
   - [ ] Scores stable with ±10 concept examples
   - [ ] Robust to different random control sets

4. **Code Quality**
   - [ ] 75%+ test coverage
   - [ ] All critical paths tested
   - [ ] Documentation complete

#### Key Tests
```python
# tests/unit/test_concept_builder.py
def test_concept_example_collection()
def test_concept_dataset_creation()
def test_random_control_generation()

# tests/unit/test_tcav_core.py
def test_cav_training()
def test_directional_derivative()
def test_tcav_score_computation()

# tests/validation/test_phase2_success.py
def test_tcav_score_clinical_plausibility()
def test_tcav_stability_across_seeds()
def test_concept_coverage()
```

---

### Phase 3: MIL Model Training ✅

#### Success Criteria
1. **Model Performance**
   - [ ] Validation AUC > 0.80 for pCR prediction
   - [ ] Balanced accuracy > 0.75
   - [ ] Calibration error < 0.1
   - [ ] Performance on external test set > 0.75 AUC

2. **Attention Quality**
   - [ ] Attention focuses on tissue (not background) > 95% of top-10 patches
   - [ ] High-attention regions correlate with pathologist annotations (IoU > 0.5)
   - [ ] Attention heatmaps visually plausible (pathologist review)

3. **Training Robustness**
   - [ ] Convergence within 50 epochs
   - [ ] No overfitting (train-val gap < 0.1 AUC)
   - [ ] Stable across 3 random seeds (std AUC < 0.05)

4. **Code Quality**
   - [ ] 80%+ test coverage
   - [ ] Model checkpointing working
   - [ ] Reproducible results

#### Key Tests
```python
# tests/unit/test_mil_model.py
def test_attention_mil_forward_pass()
def test_attention_weights_sum_to_one()
def test_model_gradient_flow()

# tests/integration/test_mil_training.py
def test_training_loop()
def test_validation_loop()
def test_checkpoint_save_load()

# tests/validation/test_phase3_success.py
def test_model_auc_threshold()
def test_attention_focus_on_tissue()
def test_training_convergence()
def test_cross_seed_stability()
```

---

### Phase 4: Interpretability Validation ✅

#### Success Criteria
1. **Faithfulness**
   - [ ] Ablation: Removing top-10% attended patches drops confidence > 20%
   - [ ] Removing random patches drops confidence < 5%
   - [ ] Masking concept regions affects predictions proportionally

2. **Stability**
   - [ ] Attention IoU > 0.7 across stain normalizations
   - [ ] Concept TCAV scores consistent across batches (std < 0.1)
   - [ ] Explanations robust to small perturbations (rotation, flip)

3. **Clinical Plausibility**
   - [ ] Pathologist agreement with explanations > 70%
   - [ ] Top concepts match known biomarkers
   - [ ] Heatmaps highlight medically relevant regions

4. **Coverage**
   - [ ] >90% of predictions have interpretable explanations
   - [ ] All high-confidence predictions (>0.8) explainable
   - [ ] Edge cases handled gracefully

#### Key Tests
```python
# tests/validation/test_faithfulness.py
def test_ablation_top_patches()
def test_ablation_random_patches()
def test_concept_perturbation()

# tests/validation/test_stability.py
def test_attention_consistency_across_stains()
def test_tcav_score_stability()
def test_perturbation_robustness()

# tests/validation/test_clinical_plausibility.py
def test_concept_alignment_with_literature()
def test_attention_on_tissue_regions()
def test_explanation_coverage()
```

---

## Test Execution

### Running Tests

```bash
# Run all tests
pytest

# Run specific category
pytest -m unit
pytest -m integration
pytest -m validation

# Run specific phase
pytest -m phase1
pytest -m phase2
pytest -m phase3

# Run success criteria tests
pytest -m success_criteria

# Run with coverage
pytest --cov=src --cov-report=html

# Run in parallel (faster)
pytest -n auto

# Run only fast tests (skip slow validation)
pytest -m "not slow"

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_hovernet_segmentation.py

# Run specific test
pytest tests/unit/test_hovernet_segmentation.py::test_segment_single_tile
```

### Continuous Integration

Tests run automatically on:
- Every commit (unit + integration tests)
- Pull requests (full test suite)
- Before merging to main (all tests + coverage check)
- Nightly builds (validation tests with real data)

### Test Data

- **Unit tests:** Use synthetic/mock data (no downloads required)
- **Integration tests:** Use small sample datasets (<100MB)
- **Validation tests:** Use full datasets (requires manual download)

Fixture data stored in `tests/fixtures/`:
```
tests/fixtures/
├── sample_tiles/          # Small H&E patches
│   ├── high_til_001.png
│   ├── necrosis_001.png
│   └── tumor_001.png
├── sample_features/       # Pre-computed features
│   └── patient_001_features.pt
├── sample_wsi/            # Tiny whole-slide image
│   └── test_slide.svs
└── expected_outputs/      # Reference outputs
    ├── segmentation_masks/
    └── attention_maps/
```

---

## Success Criteria Checklist

Before moving to the next phase, ALL criteria must be met:

### Phase 1: HoverNet Segmentation
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Dice score > 0.75
- [ ] TIL correlation > 0.7
- [ ] Processing speed < 2s/tile
- [ ] Code review completed
- [ ] Documentation updated

### Phase 2: TCAV Integration
- [ ] All unit tests passing
- [ ] 8+ concepts defined
- [ ] TCAV scores computed
- [ ] Clinical plausibility validated
- [ ] Stability tests passed
- [ ] Code review completed
- [ ] Documentation updated

### Phase 3: MIL Model
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] AUC > 0.80
- [ ] Attention quality validated
- [ ] Training stability confirmed
- [ ] Model saved and loadable
- [ ] Code review completed
- [ ] Documentation updated

### Phase 4: Interpretability
- [ ] All validation tests passing
- [ ] Faithfulness tests passed
- [ ] Stability tests passed
- [ ] Pathologist review completed
- [ ] >90% explanation coverage
- [ ] Final report generated
- [ ] Code review completed
- [ ] Documentation updated

---

## Test Maintenance

### Regular Tasks
- **Weekly:** Review failing tests, update fixtures
- **Monthly:** Review test coverage, add missing tests
- **Per Phase:** Update success criteria, validate metrics

### Test Hygiene
- Remove obsolete tests
- Keep tests independent
- Use descriptive test names
- Document complex test logic
- Maintain test fixtures

### Performance
- Monitor test suite execution time
- Optimize slow tests
- Use pytest-xdist for parallelization
- Cache expensive operations

---

## Metrics Dashboard

Track these metrics throughout development:

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Code Coverage** | >80% | TBD | 🔴 |
| **Test Pass Rate** | 100% | TBD | 🔴 |
| **Phase 1 Success** | All criteria | 0/7 | 🔴 |
| **Phase 2 Success** | All criteria | 0/7 | 🔴 |
| **Phase 3 Success** | All criteria | 0/7 | 🔴 |
| **Phase 4 Success** | All criteria | 0/7 | 🔴 |
| **Model AUC** | >0.80 | TBD | 🔴 |
| **Pathologist Approval** | >70% | TBD | 🔴 |

Legend: 🔴 Not started | 🟡 In progress | 🟢 Completed

---

## Resources

- **pytest docs:** https://docs.pytest.org/
- **pytest-cov:** https://pytest-cov.readthedocs.io/
- **TDD principles:** https://testdriven.io/
- **Python testing best practices:** https://realpython.com/pytest-python-testing/

---

**Next Steps:**
1. Review this testing strategy
2. Set up pytest environment
3. Write first test (RED)
4. Implement feature (GREEN)
5. Refactor (REFACTOR)
6. Repeat!
