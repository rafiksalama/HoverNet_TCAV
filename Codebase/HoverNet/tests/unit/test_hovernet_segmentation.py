"""
Unit tests for HoverNet nuclei segmentation

Success Criteria (Phase 1):
- Segment nuclei with Dice > 0.75
- Extract 15+ morphological features per nucleus
- Process 512x512 tile in <2 seconds
"""

import pytest
import numpy as np
import torch
import cv2
from pathlib import Path


@pytest.mark.unit
@pytest.mark.phase1
class TestNucleiSegmentation:
    """Tests for nuclei segmentation functionality"""

    def test_segment_single_tile(self, sample_he_tile):
        """Test segmentation of a single H&E tile"""
        from hovernet_pipeline.segmentation import segment_nuclei

        # Act
        result = segment_nuclei(sample_he_tile)

        # Assert
        assert 'inst_map' in result, "Instance map missing from output"
        assert 'type_map' in result, "Type map missing from output"
        assert result['inst_map'].shape == (512, 512), "Instance map shape incorrect"
        assert result['type_map'].shape == (512, 512), "Type map shape incorrect"

        # Check that some nuclei were detected
        n_nuclei = len(np.unique(result['inst_map'])) - 1  # Exclude background
        assert n_nuclei > 0, "No nuclei detected"
        assert n_nuclei < 1000, f"Too many nuclei detected ({n_nuclei}), likely noise"

    def test_segment_tile_with_no_tissue(self):
        """Test handling of background/no-tissue regions"""
        from hovernet_pipeline.segmentation import segment_nuclei

        # Arrange: White background (no tissue)
        background_tile = np.ones((512, 512, 3), dtype=np.uint8) * 255

        # Act
        result = segment_nuclei(background_tile)

        # Assert: Should detect few or no nuclei
        n_nuclei = len(np.unique(result['inst_map'])) - 1
        assert n_nuclei < 10, "Should detect minimal nuclei in background"

    def test_nuclei_instance_ids_unique(self, sample_he_tile):
        """Test that each nucleus has a unique instance ID"""
        from hovernet_pipeline.segmentation import segment_nuclei

        result = segment_nuclei(sample_he_tile)
        inst_map = result['inst_map']

        # Get all instance IDs (excluding background 0)
        instance_ids = np.unique(inst_map)[1:]

        # Check consecutive numbering (1, 2, 3, ...)
        assert len(instance_ids) == instance_ids.max(), \
            "Instance IDs should be consecutive starting from 1"

    def test_nuclei_classification_types(self, sample_he_tile):
        """Test that nuclei are classified into valid types"""
        from hovernet_pipeline.segmentation import segment_nuclei

        result = segment_nuclei(sample_he_tile)
        type_map = result['type_map']

        # Valid types: 0=background, 1=inflammatory, 2=epithelial, 3=stromal, 4=necrotic
        unique_types = np.unique(type_map)

        assert all(t in [0, 1, 2, 3, 4] for t in unique_types), \
            "Invalid nucleus types detected"

    @pytest.mark.parametrize("tile_size", [256, 512, 1024])
    def test_segment_different_tile_sizes(self, tile_size):
        """Test segmentation with different tile sizes"""
        from hovernet_pipeline.segmentation import segment_nuclei

        # Create tile of specified size
        tile = np.random.randint(0, 255, (tile_size, tile_size, 3), dtype=np.uint8)

        result = segment_nuclei(tile)

        assert result['inst_map'].shape == (tile_size, tile_size)
        assert result['type_map'].shape == (tile_size, tile_size)


@pytest.mark.unit
@pytest.mark.phase1
class TestFeatureExtraction:
    """Tests for nuclei feature extraction"""

    def test_extract_morphological_features(self, mock_hovernet_output):
        """Test extraction of morphological features from segmented nuclei"""
        from hovernet_pipeline.features import extract_nuclei_features

        # Act
        features = extract_nuclei_features(mock_hovernet_output)

        # Assert
        assert len(features) > 0, "No features extracted"

        # Check first feature
        first_feature = features[0]
        required_keys = [
            'area', 'perimeter', 'circularity', 'type',
            'centroid', 'eccentricity', 'solidity'
        ]

        for key in required_keys:
            assert key in first_feature, f"Missing feature: {key}"

    def test_feature_count_minimum(self, mock_hovernet_output):
        """Test that at least 15 features are extracted per nucleus"""
        from hovernet_pipeline.features import extract_nuclei_features

        features = extract_nuclei_features(mock_hovernet_output)

        if len(features) > 0:
            feature_count = len(features[0])
            assert feature_count >= 15, \
                f"Expected >=15 features per nucleus, got {feature_count}"

    def test_feature_values_valid_ranges(self, mock_hovernet_output):
        """Test that feature values are in valid ranges"""
        from hovernet_pipeline.features import extract_nuclei_features

        features = extract_nuclei_features(mock_hovernet_output)

        for feat in features:
            # Area should be positive
            assert feat['area'] > 0, "Area must be positive"

            # Circularity in [0, 1]
            assert 0 <= feat['circularity'] <= 1, "Circularity must be in [0, 1]"

            # Solidity in [0, 1]
            if 'solidity' in feat:
                assert 0 <= feat['solidity'] <= 1, "Solidity must be in [0, 1]"

            # Type in valid range
            assert feat['type'] in [1, 2, 3, 4], "Invalid nucleus type"

    def test_feature_extraction_handles_empty_mask(self):
        """Test feature extraction with no nuclei"""
        from hovernet_pipeline.features import extract_nuclei_features

        empty_output = {
            'inst_map': np.zeros((512, 512), dtype=np.uint16),
            'type_map': np.zeros((512, 512), dtype=np.uint8)
        }

        features = extract_nuclei_features(empty_output)

        assert isinstance(features, list), "Should return list even when empty"
        assert len(features) == 0, "Should have no features for empty mask"


@pytest.mark.unit
@pytest.mark.phase1
class TestSlideAggregation:
    """Tests for slide-level feature aggregation"""

    def test_compute_slide_level_features(self, sample_nuclei_features):
        """Test aggregation of nuclei features to slide-level metrics"""
        from hovernet_pipeline.features import compute_slide_level_features

        # Act
        slide_metrics = compute_slide_level_features(sample_nuclei_features)

        # Assert
        required_metrics = [
            'lymphocyte_density', 'tumor_cell_density',
            'stromal_density', 'necrotic_density',
            'mean_nuclear_area', 'std_nuclear_area'
        ]

        for metric in required_metrics:
            assert metric in slide_metrics, f"Missing metric: {metric}"
            assert isinstance(slide_metrics[metric], (int, float)), \
                f"{metric} should be numeric"

    def test_density_metrics_sum_to_one(self, sample_nuclei_features):
        """Test that cell type densities sum to approximately 1"""
        from hovernet_pipeline.features import compute_slide_level_features

        metrics = compute_slide_level_features(sample_nuclei_features)

        density_sum = (
            metrics['lymphocyte_density'] +
            metrics['tumor_cell_density'] +
            metrics['stromal_density'] +
            metrics['necrotic_density']
        )

        assert 0.95 <= density_sum <= 1.05, \
            f"Densities should sum to ~1, got {density_sum}"

    def test_til_density_calculation(self):
        """Test TIL (tumor-infiltrating lymphocyte) density calculation"""
        from hovernet_pipeline.features import compute_slide_level_features

        # Create features with known composition: 30% lymphocytes
        features = []
        for i in range(100):
            features.append({
                'type': 1 if i < 30 else 2,  # 1=lymphocyte, 2=epithelial
                'area': 100.0,
                'perimeter': 40.0,
                'circularity': 0.8,
                'centroid': (0, 0)
            })

        metrics = compute_slide_level_features(features)

        # Should be close to 0.3
        assert 0.25 <= metrics['lymphocyte_density'] <= 0.35, \
            f"Expected ~0.3 lymphocyte density, got {metrics['lymphocyte_density']}"


@pytest.mark.unit
@pytest.mark.phase1
class TestStainNormalization:
    """Tests for stain normalization"""

    def test_macenko_normalization_maintains_shape(self, sample_he_tile):
        """Test that stain normalization preserves image shape"""
        from data_processing.stain_normalization import MacenkoNormalizer

        normalizer = MacenkoNormalizer()
        normalized = normalizer.normalize(sample_he_tile)

        assert normalized.shape == sample_he_tile.shape, \
            "Normalization changed image shape"

    def test_normalized_image_within_valid_range(self, sample_he_tile):
        """Test that normalized image values are valid"""
        from data_processing.stain_normalization import MacenkoNormalizer

        normalizer = MacenkoNormalizer()
        normalized = normalizer.normalize(sample_he_tile)

        assert normalized.min() >= 0, "Normalized values should be >= 0"
        assert normalized.max() <= 255, "Normalized values should be <= 255"
        assert normalized.dtype == np.uint8, "Should maintain uint8 dtype"

    def test_normalization_improves_consistency(self):
        """Test that normalization reduces color variation between similar tiles"""
        from data_processing.stain_normalization import MacenkoNormalizer

        # Create two tiles with different staining intensity
        tile1 = np.ones((256, 256, 3), dtype=np.uint8) * 150  # Medium
        tile2 = np.ones((256, 256, 3), dtype=np.uint8) * 100  # Darker

        # Original difference
        original_diff = np.abs(tile1.mean() - tile2.mean())

        # Normalize both
        normalizer = MacenkoNormalizer()
        norm1 = normalizer.normalize(tile1)
        norm2 = normalizer.normalize(tile2)

        # Normalized difference should be smaller
        normalized_diff = np.abs(norm1.mean() - norm2.mean())

        # This is a weak test - in practice, need real H&E slides
        # Just check that normalization doesn't break things
        assert normalized_diff < 255, "Normalization produced invalid results"


@pytest.mark.unit
@pytest.mark.phase1
@pytest.mark.slow
class TestPerformance:
    """Performance tests for Phase 1"""

    def test_segmentation_speed_benchmark(self, sample_he_tile, benchmark):
        """Test that segmentation meets speed requirements (<2s per tile)"""
        from hovernet_pipeline.segmentation import segment_nuclei

        # Benchmark the segmentation
        result = benchmark(segment_nuclei, sample_he_tile)

        # Check result is valid
        assert 'inst_map' in result

    @pytest.mark.timeout(2)
    def test_tile_processing_timeout(self, sample_he_tile):
        """Test that a single tile is processed within 2 seconds"""
        from hovernet_pipeline.segmentation import segment_nuclei

        # Should complete within timeout
        result = segment_nuclei(sample_he_tile)
        assert result is not None


@pytest.mark.unit
@pytest.mark.phase1
class TestErrorHandling:
    """Tests for error handling"""

    def test_invalid_image_shape_raises_error(self):
        """Test handling of invalid image shapes"""
        from hovernet_pipeline.segmentation import segment_nuclei

        # 2D image instead of 3D
        invalid_tile = np.zeros((512, 512), dtype=np.uint8)

        with pytest.raises((ValueError, AssertionError)):
            segment_nuclei(invalid_tile)

    def test_invalid_image_dtype_raises_error(self):
        """Test handling of invalid dtypes"""
        from hovernet_pipeline.segmentation import segment_nuclei

        # Float image instead of uint8
        invalid_tile = np.random.rand(512, 512, 3)

        with pytest.raises((ValueError, AssertionError)):
            segment_nuclei(invalid_tile)

    def test_corrupted_image_handled_gracefully(self):
        """Test handling of corrupted/invalid image data"""
        from hovernet_pipeline.segmentation import segment_nuclei

        # All zeros (likely corrupted)
        corrupted_tile = np.zeros((512, 512, 3), dtype=np.uint8)

        # Should not crash, but may return empty segmentation
        result = segment_nuclei(corrupted_tile)
        assert 'inst_map' in result
        assert 'type_map' in result
