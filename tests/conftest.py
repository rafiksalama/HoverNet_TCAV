"""
Pytest configuration and shared fixtures
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import tempfile
import shutil


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def temp_dir():
    """Temporary directory for test outputs"""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture
def sample_he_tile():
    """Generate a synthetic H&E tile (512x512x3)"""
    # Create realistic H&E color distribution
    tile = np.zeros((512, 512, 3), dtype=np.uint8)

    # Background (light pink)
    tile[:, :] = [235, 210, 220]

    # Add some nuclei (dark purple/blue)
    np.random.seed(42)
    for _ in range(50):
        x, y = np.random.randint(50, 462, size=2)
        radius = np.random.randint(5, 15)

        # Create circular nucleus
        yy, xx = np.ogrid[:512, :512]
        mask = (xx - x)**2 + (yy - y)**2 <= radius**2

        # Hematoxylin color (purple/blue)
        tile[mask] = [100, 80, 130]

    # Add some eosin-stained cytoplasm (pink regions)
    for _ in range(30):
        x, y = np.random.randint(50, 462, size=2)
        width, height = np.random.randint(20, 50, size=2)
        tile[y:y+height, x:x+width] = [220, 150, 180]

    return tile


@pytest.fixture
def sample_he_tile_pil(sample_he_tile):
    """PIL Image version of sample H&E tile"""
    return Image.fromarray(sample_he_tile)


@pytest.fixture
def sample_nuclei_mask():
    """Generate sample nuclei segmentation mask"""
    mask = np.zeros((512, 512), dtype=np.uint16)

    np.random.seed(42)
    nucleus_id = 1
    for _ in range(50):
        x, y = np.random.randint(50, 462, size=2)
        radius = np.random.randint(5, 15)

        yy, xx = np.ogrid[:512, :512]
        circle = (xx - x)**2 + (yy - y)**2 <= radius**2

        # Avoid overlap
        if np.sum(mask[circle] > 0) < 5:
            mask[circle] = nucleus_id
            nucleus_id += 1

    return mask


@pytest.fixture
def sample_type_map():
    """Generate sample nuclei type classification map"""
    type_map = np.zeros((512, 512), dtype=np.uint8)

    # Types: 0=background, 1=inflammatory, 2=epithelial, 3=stromal, 4=necrotic
    np.random.seed(42)
    for _ in range(50):
        x, y = np.random.randint(50, 462, size=2)
        radius = np.random.randint(5, 15)

        yy, xx = np.ogrid[:512, :512]
        circle = (xx - x)**2 + (yy - y)**2 <= radius**2

        # Random type
        nuc_type = np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1])
        type_map[circle] = nuc_type

    return type_map


@pytest.fixture
def sample_nuclei_features():
    """Generate sample nuclei morphological features"""
    np.random.seed(42)
    n_nuclei = 50

    features = []
    for i in range(n_nuclei):
        features.append({
            'id': i + 1,
            'area': np.random.uniform(50, 200),
            'perimeter': np.random.uniform(25, 60),
            'circularity': np.random.uniform(0.6, 0.95),
            'type': np.random.choice([1, 2, 3, 4]),
            'centroid': (np.random.randint(0, 512), np.random.randint(0, 512)),
            'eccentricity': np.random.uniform(0.3, 0.9),
            'solidity': np.random.uniform(0.85, 0.99)
        })

    return features


@pytest.fixture
def sample_patch_features():
    """Generate sample patch feature embeddings"""
    np.random.seed(42)
    n_patches = 100
    feature_dim = 1024

    features = torch.randn(n_patches, feature_dim)
    # Normalize
    features = features / features.norm(dim=1, keepdim=True)

    return features


@pytest.fixture
def sample_patch_coords():
    """Generate sample patch coordinates"""
    np.random.seed(42)
    n_patches = 100

    coords = []
    for i in range(n_patches):
        x = (i % 10) * 512
        y = (i // 10) * 512
        coords.append((x, y))

    return coords


@pytest.fixture
def sample_wsi_manifest():
    """Generate sample WSI dataset manifest"""
    import pandas as pd

    data = {
        'patient_id': [f'patient_{i:03d}' for i in range(1, 11)],
        'slide_path': [f'data/raw/patient_{i:03d}.svs' for i in range(1, 11)],
        'pCR': [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
        'HER2_status': [1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
        'ER_status': [0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
        'PR_status': [0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
    }

    return pd.DataFrame(data)


@pytest.fixture
def mock_hovernet_output(sample_nuclei_mask, sample_type_map):
    """Mock HoverNet segmentation output"""
    return {
        'inst_map': sample_nuclei_mask,
        'type_map': sample_type_map,
        'inst_type': {
            i: sample_type_map[sample_nuclei_mask == i][0]
            for i in range(1, sample_nuclei_mask.max() + 1)
            if np.sum(sample_nuclei_mask == i) > 0
        }
    }


@pytest.fixture
def sample_concept_images(test_data_dir):
    """Generate sample concept image sets"""
    concept_dir = test_data_dir / "concepts"
    concept_dir.mkdir(parents=True, exist_ok=True)

    concepts = ['high_TILs', 'low_TILs', 'necrosis', 'viable_tumor']

    for concept in concepts:
        (concept_dir / concept).mkdir(exist_ok=True)

        # Generate 10 example images per concept
        for i in range(10):
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

            # Add concept-specific patterns
            if concept == 'high_TILs':
                # Many small dark nuclei
                for _ in range(100):
                    x, y = np.random.randint(10, 246, size=2)
                    img[y-3:y+3, x-3:x+3] = [80, 70, 120]

            elif concept == 'necrosis':
                # Large pink/red areas
                img[:, :] = [200, 100, 120]

            elif concept == 'viable_tumor':
                # Dense nuclei with variation
                for _ in range(50):
                    x, y = np.random.randint(10, 246, size=2)
                    r = np.random.randint(5, 12)
                    img[y-r:y+r, x-r:x+r] = [90, 80, 100]

            Image.fromarray(img).save(concept_dir / concept / f"{i:03d}.png")

    return concept_dir


@pytest.fixture
def mock_tcav_scores():
    """Mock TCAV concept importance scores"""
    return {
        'high_TILs': 0.85,
        'low_TILs': 0.20,
        'geographic_necrosis': 0.72,
        'viable_tumor': 0.35,
        'fibrosis': 0.68,
        'high_mitosis': 0.45,
        'low_mitosis': 0.30,
        'poor_differentiation': 0.55
    }


@pytest.fixture
def sample_attention_weights():
    """Generate sample attention weights"""
    np.random.seed(42)
    n_patches = 100

    # Generate attention with some high-attention patches
    weights = np.random.exponential(scale=0.3, size=n_patches)
    weights = weights / weights.sum()  # Normalize to sum to 1

    return torch.tensor(weights, dtype=torch.float32)


@pytest.fixture
def trained_mil_model():
    """Mock trained MIL model"""
    from mil_model.attention_mil import AttentionMIL

    model = AttentionMIL(
        feature_dim=1024,
        hidden_dim=256,
        n_classes=2,
        dropout=0.25
    )

    # Initialize with some reasonable weights
    torch.manual_seed(42)
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)

    return model


@pytest.fixture(scope="session")
def expected_metrics():
    """Expected performance metrics for validation"""
    return {
        'phase1': {
            'dice_score': 0.75,
            'til_correlation': 0.70,
            'processing_speed': 2.0,  # seconds per tile
        },
        'phase2': {
            'min_tcav_score_til': 0.60,
            'max_tcav_score_tumor': 0.40,
            'tcav_std_threshold': 0.10,
        },
        'phase3': {
            'min_auc': 0.80,
            'min_balanced_accuracy': 0.75,
            'max_calibration_error': 0.10,
            'max_overfitting_gap': 0.10,
        },
        'phase4': {
            'min_ablation_drop': 0.20,
            'max_random_ablation_drop': 0.05,
            'min_attention_iou': 0.70,
            'min_pathologist_agreement': 0.70,
        }
    }


# Markers for test organization
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "validation: Validation and success criteria tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take >5 seconds"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "data: Tests requiring downloaded datasets"
    )
    config.addinivalue_line(
        "markers", "phase1: Phase 1 HoverNet tests"
    )
    config.addinivalue_line(
        "markers", "phase2: Phase 2 TCAV tests"
    )
    config.addinivalue_line(
        "markers", "phase3: Phase 3 MIL model tests"
    )
    config.addinivalue_line(
        "markers", "success_criteria: Success criteria validation tests"
    )
