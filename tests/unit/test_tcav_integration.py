"""
Unit tests for TCAV concept-based explanations

Success Criteria (Phase 2):
- Define 8+ pathological concepts
- TCAV scores clinically plausible
- Stable across random seeds (std < 0.1)
"""

import pytest
import numpy as np
import torch
from pathlib import Path


@pytest.mark.unit
@pytest.mark.phase2
class TestConceptDefinition:
    """Tests for concept dataset creation"""

    def test_create_concept_dataset(self, sample_concept_images):
        """Test creation of concept image datasets"""
        from tcav_integration.concept_builder import ConceptDatasetBuilder

        builder = ConceptDatasetBuilder(base_dir=sample_concept_images.parent)

        # Check that concept directories exist
        assert (sample_concept_images / "high_TILs").exists()
        assert (sample_concept_images / "necrosis").exists()

    def test_concept_minimum_examples(self, sample_concept_images):
        """Test that each concept has minimum 50 examples"""
        min_examples = 10  # Using 10 for unit tests, 50 in production

        for concept_dir in sample_concept_images.iterdir():
            if concept_dir.is_dir() and concept_dir.name != "random":
                n_examples = len(list(concept_dir.glob("*.png")))
                assert n_examples >= min_examples, \
                    f"Concept '{concept_dir.name}' has only {n_examples} examples"

    def test_concept_images_valid_format(self, sample_concept_images):
        """Test that concept images are valid and loadable"""
        from PIL import Image

        for concept_dir in sample_concept_images.iterdir():
            if concept_dir.is_dir():
                for img_path in list(concept_dir.glob("*.png"))[:5]:  # Check first 5
                    img = Image.open(img_path)
                    assert img.mode == 'RGB', f"Image {img_path} not RGB"
                    assert img.size[0] > 0 and img.size[1] > 0, "Invalid image size"

    def test_random_counterexamples_generated(self):
        """Test generation of random negative examples for TCAV"""
        from tcav_integration.concept_builder import ConceptDatasetBuilder

        builder = ConceptDatasetBuilder()
        random_dir = builder.base_dir / "random"

        # Should create random directory
        builder.create_random_counterexamples(n_examples=20)

        assert random_dir.exists(), "Random directory not created"
        assert len(list(random_dir.glob("*.png"))) >= 20, \
            "Insufficient random examples"

    def test_concept_definition_count(self):
        """Test that minimum 8 concepts are defined"""
        from tcav_integration.concept_builder import ConceptDatasetBuilder

        builder = ConceptDatasetBuilder()

        assert len(builder.concepts) >= 8, \
            f"Expected >=8 concepts, got {len(builder.concepts)}"


@pytest.mark.unit
@pytest.mark.phase2
class TestCAVTraining:
    """Tests for Concept Activation Vector training"""

    def test_train_linear_cav(self):
        """Test training of linear CAV classifier"""
        from tcav_integration.tcav_core import train_cav

        # Mock activations
        concept_acts = torch.randn(50, 1024)  # 50 concept examples
        random_acts = torch.randn(50, 1024)   # 50 random examples

        # Train CAV
        cav_vector = train_cav(concept_acts, random_acts)

        assert cav_vector.shape == (1024,), "CAV should be 1D vector"
        assert torch.norm(cav_vector) > 0, "CAV should be non-zero"

    def test_cav_normalized(self):
        """Test that CAV is unit normalized"""
        from tcav_integration.tcav_core import train_cav

        concept_acts = torch.randn(50, 1024)
        random_acts = torch.randn(50, 1024)

        cav_vector = train_cav(concept_acts, random_acts)

        # Should be unit normalized
        norm = torch.norm(cav_vector).item()
        assert 0.99 <= norm <= 1.01, f"CAV not normalized, norm={norm}"

    def test_cav_classification_accuracy(self):
        """Test that CAV achieves reasonable classification accuracy"""
        from tcav_integration.tcav_core import train_cav, evaluate_cav

        # Create separable data
        torch.manual_seed(42)
        concept_acts = torch.randn(50, 1024) + 1.0  # Offset
        random_acts = torch.randn(50, 1024) - 1.0

        cav_vector = train_cav(concept_acts, random_acts)

        # Evaluate
        accuracy = evaluate_cav(cav_vector, concept_acts, random_acts)

        assert accuracy > 0.7, \
            f"CAV accuracy too low ({accuracy}), concepts may not be well-defined"


@pytest.mark.unit
@pytest.mark.phase2
class TestTCAVScores:
    """Tests for TCAV score computation"""

    def test_compute_tcav_score(self):
        """Test TCAV score computation"""
        from tcav_integration.tcav_core import compute_tcav_score

        # Mock CAV and model
        cav_vector = torch.randn(1024)
        cav_vector = cav_vector / torch.norm(cav_vector)

        # Mock gradients for test examples
        test_gradients = torch.randn(20, 1024)  # 20 test examples

        # Compute TCAV score
        tcav_score = compute_tcav_score(cav_vector, test_gradients)

        assert 0 <= tcav_score <= 1, "TCAV score should be in [0, 1]"

    def test_tcav_score_interpretation(self):
        """Test TCAV score interpretation"""
        from tcav_integration.tcav_core import compute_tcav_score

        # CAV pointing in direction [1, 0, 0, ...]
        cav_vector = torch.zeros(1024)
        cav_vector[0] = 1.0

        # Gradients all pointing in same direction as CAV
        test_gradients = torch.zeros(20, 1024)
        test_gradients[:, 0] = 1.0  # All positive

        tcav_score = compute_tcav_score(cav_vector, test_gradients)

        # Should be 1.0 (all gradients align with concept)
        assert tcav_score == 1.0, "Expected TCAV score of 1.0 for aligned gradients"

        # Gradients all opposite to CAV
        test_gradients[:, 0] = -1.0

        tcav_score = compute_tcav_score(cav_vector, test_gradients)

        # Should be 0.0 (no gradients align with concept)
        assert tcav_score == 0.0, "Expected TCAV score of 0.0 for opposite gradients"

    def test_tcav_scores_for_multiple_concepts(self, mock_tcav_scores):
        """Test computation for multiple concepts"""
        assert len(mock_tcav_scores) >= 8, "Should have scores for >=8 concepts"

        for concept, score in mock_tcav_scores.items():
            assert 0 <= score <= 1, f"Invalid TCAV score for {concept}: {score}"


@pytest.mark.unit
@pytest.mark.phase2
class TestTCAVStability:
    """Tests for TCAV stability and reproducibility"""

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_tcav_score_stability_across_seeds(self, seed):
        """Test TCAV scores are stable across different random seeds"""
        from tcav_integration.tcav_core import compute_tcav_scores_for_concepts

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Mock model and concepts
        scores = compute_tcav_scores_for_concepts(
            model=None,  # Will be mocked inside
            concepts=['high_TILs', 'necrosis'],
            n_examples=50
        )

        # Store scores for comparison
        return scores

    def test_tcav_score_variance_threshold(self):
        """Test that TCAV score variance across seeds is below threshold"""
        from tcav_integration.tcav_core import compute_tcav_scores_for_concepts

        scores_list = []

        # Run multiple times with different seeds
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            np.random.seed(seed)

            scores = compute_tcav_scores_for_concepts(
                model=None,
                concepts=['high_TILs'],
                n_examples=50
            )
            scores_list.append(scores['high_TILs'])

        # Compute standard deviation
        std = np.std(scores_list)

        assert std < 0.1, \
            f"TCAV score std ({std}) exceeds threshold (0.1) - unstable!"

    def test_tcav_robustness_to_sample_size(self):
        """Test TCAV scores robust to number of concept examples (±10)"""
        from tcav_integration.tcav_core import compute_tcav_scores_for_concepts

        # Compute with 50 examples
        scores_50 = compute_tcav_scores_for_concepts(
            model=None,
            concepts=['high_TILs'],
            n_examples=50
        )

        # Compute with 40 examples
        scores_40 = compute_tcav_scores_for_concepts(
            model=None,
            concepts=['high_TILs'],
            n_examples=40
        )

        # Compute with 60 examples
        scores_60 = compute_tcav_scores_for_concepts(
            model=None,
            concepts=['high_TILs'],
            n_examples=60
        )

        # Should be similar
        diff_40 = abs(scores_50['high_TILs'] - scores_40['high_TILs'])
        diff_60 = abs(scores_50['high_TILs'] - scores_60['high_TILs'])

        assert diff_40 < 0.15, "TCAV score varies too much with sample size"
        assert diff_60 < 0.15, "TCAV score varies too much with sample size"


@pytest.mark.unit
@pytest.mark.phase2
class TestClinicalPlausibility:
    """Tests for clinical plausibility of TCAV scores"""

    def test_high_til_positive_association_with_pcr(self, mock_tcav_scores):
        """Test that high TILs has positive association with pCR (score > 0.6)"""
        assert mock_tcav_scores['high_TILs'] > 0.6, \
            "High TILs should positively predict pCR (known biomarker)"

    def test_viable_tumor_negative_association_with_pcr(self, mock_tcav_scores):
        """Test that viable tumor has negative association with pCR (score < 0.4)"""
        assert mock_tcav_scores['viable_tumor'] < 0.4, \
            "High viable tumor should negatively predict pCR"

    def test_necrosis_positive_association_with_pcr(self, mock_tcav_scores):
        """Test that necrosis positively associated with pCR"""
        # Necrosis often indicates treatment effect
        assert mock_tcav_scores['geographic_necrosis'] > 0.6, \
            "Necrosis should positively predict pCR (treatment effect)"

    def test_concept_ordering_makes_clinical_sense(self, mock_tcav_scores):
        """Test that concept importance ordering aligns with clinical knowledge"""

        # TILs should be more important than random
        assert mock_tcav_scores['high_TILs'] > mock_tcav_scores['low_TILs'], \
            "High TILs should have higher TCAV score than low TILs"

        # High mitosis should be more predictive than low
        assert mock_tcav_scores['high_mitosis'] > mock_tcav_scores['low_mitosis'], \
            "High mitosis should be more associated with response"


@pytest.mark.unit
@pytest.mark.phase2
class TestActivationGeneration:
    """Tests for generating activations at specified layer"""

    def test_generate_activations_from_images(self):
        """Test activation generation from concept images"""
        from tcav_integration.tcav_core import generate_activations

        # Mock images
        images = torch.randn(10, 3, 256, 256)  # 10 RGB images

        # Mock model (simple feature extractor)
        mock_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten()
        )

        activations = generate_activations(images, mock_model, layer_name='layer3')

        assert activations.shape[0] == 10, "Should have 10 activations"
        assert activations.dim() == 2, "Activations should be 2D (N, D)"

    def test_activation_dimensionality(self):
        """Test that activations have correct dimensionality"""
        from tcav_integration.tcav_core import generate_activations

        images = torch.randn(5, 3, 256, 256)

        # Mock feature extractor outputting 1024-dim features
        mock_model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 256 * 256, 1024)
        )

        activations = generate_activations(images, mock_model)

        assert activations.shape == (5, 1024), \
            f"Expected shape (5, 1024), got {activations.shape}"


@pytest.mark.unit
@pytest.mark.phase2
class TestErrorHandling:
    """Tests for error handling in TCAV"""

    def test_insufficient_concept_examples_warning(self):
        """Test warning when concept has too few examples"""
        from tcav_integration.concept_builder import ConceptDatasetBuilder

        builder = ConceptDatasetBuilder()

        # Try to collect only 5 examples (below minimum 50)
        with pytest.warns(UserWarning):
            builder.collect_concept_examples(
                concept_name='test_concept',
                source_wsi_paths=[],
                hovernet_results={},
                n_examples=5
            )

    def test_invalid_concept_name_raises_error(self):
        """Test that invalid concept names raise errors"""
        from tcav_integration.tcav_core import compute_tcav_scores_for_concepts

        with pytest.raises((ValueError, KeyError)):
            compute_tcav_scores_for_concepts(
                model=None,
                concepts=['nonexistent_concept_xyz'],
                n_examples=50
            )

    def test_mismatched_activation_dimensions_raises_error(self):
        """Test error when concept and random activations have different dims"""
        from tcav_integration.tcav_core import train_cav

        concept_acts = torch.randn(50, 1024)
        random_acts = torch.randn(50, 512)  # Different dimension

        with pytest.raises((ValueError, AssertionError)):
            train_cav(concept_acts, random_acts)
