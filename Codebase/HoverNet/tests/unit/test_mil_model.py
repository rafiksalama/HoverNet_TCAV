"""
Unit tests for MIL (Multiple Instance Learning) model

Success Criteria (Phase 3):
- Validation AUC > 0.80
- Attention focuses on tissue (not background)
- Training converges within 50 epochs
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


@pytest.mark.unit
@pytest.mark.phase3
class TestAttentionMIL:
    """Tests for Attention-based MIL architecture"""

    def test_model_initialization(self):
        """Test model can be initialized with correct architecture"""
        from mil_model.attention_mil import AttentionMIL

        model = AttentionMIL(
            feature_dim=1024,
            hidden_dim=256,
            n_classes=2,
            dropout=0.25
        )

        assert model is not None
        assert hasattr(model, 'attention_net')
        assert hasattr(model, 'classifier')

    def test_forward_pass_shape(self, sample_patch_features):
        """Test forward pass produces correct output shapes"""
        from mil_model.attention_mil import AttentionMIL

        model = AttentionMIL(feature_dim=1024, hidden_dim=256, n_classes=2)

        # Forward pass
        logits, attention_weights = model(sample_patch_features)

        assert logits.shape == (2,), f"Expected logits shape (2,), got {logits.shape}"
        assert attention_weights.shape == (100,), \
            f"Expected attention shape (100,), got {attention_weights.shape}"

    def test_attention_weights_sum_to_one(self, sample_patch_features):
        """Test that attention weights sum to 1 (probability distribution)"""
        from mil_model.attention_mil import AttentionMIL

        model = AttentionMIL(feature_dim=1024, hidden_dim=256, n_classes=2)

        _, attention_weights = model(sample_patch_features)

        attention_sum = attention_weights.sum().item()

        assert 0.99 <= attention_sum <= 1.01, \
            f"Attention weights should sum to 1, got {attention_sum}"

    def test_attention_weights_all_positive(self, sample_patch_features):
        """Test that all attention weights are positive"""
        from mil_model.attention_mil import AttentionMIL

        model = AttentionMIL(feature_dim=1024, hidden_dim=256, n_classes=2)

        _, attention_weights = model(sample_patch_features)

        assert (attention_weights >= 0).all(), "All attention weights should be >= 0"

    def test_model_gradient_flow(self, sample_patch_features):
        """Test that gradients flow through the model"""
        from mil_model.attention_mil import AttentionMIL

        model = AttentionMIL(feature_dim=1024, hidden_dim=256, n_classes=2)

        # Forward pass
        logits, _ = model(sample_patch_features)
        loss = logits[0]  # Dummy loss

        # Backward pass
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_model_deterministic_with_seed(self, sample_patch_features):
        """Test that model produces deterministic outputs with fixed seed"""
        from mil_model.attention_mil import AttentionMIL

        torch.manual_seed(42)
        model1 = AttentionMIL(feature_dim=1024, hidden_dim=256, n_classes=2)
        logits1, attn1 = model1(sample_patch_features)

        torch.manual_seed(42)
        model2 = AttentionMIL(feature_dim=1024, hidden_dim=256, n_classes=2)
        logits2, attn2 = model2(sample_patch_features)

        assert torch.allclose(logits1, logits2), "Model not deterministic"
        assert torch.allclose(attn1, attn2), "Attention not deterministic"

    @pytest.mark.parametrize("n_patches", [10, 100, 1000])
    def test_model_handles_variable_bag_sizes(self, n_patches):
        """Test model works with different numbers of patches"""
        from mil_model.attention_mil import AttentionMIL

        model = AttentionMIL(feature_dim=1024, hidden_dim=256, n_classes=2)

        features = torch.randn(n_patches, 1024)
        logits, attention = model(features)

        assert logits.shape == (2,)
        assert attention.shape == (n_patches,)

    def test_model_save_load(self, trained_mil_model, temp_dir):
        """Test saving and loading model weights"""
        save_path = temp_dir / "model.pth"

        # Save
        torch.save(trained_mil_model.state_dict(), save_path)

        # Load into new model
        from mil_model.attention_mil import AttentionMIL
        loaded_model = AttentionMIL(feature_dim=1024, hidden_dim=256, n_classes=2)
        loaded_model.load_state_dict(torch.load(save_path))

        # Compare parameters
        for (n1, p1), (n2, p2) in zip(
            trained_mil_model.named_parameters(),
            loaded_model.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2), f"Parameter {n1} not loaded correctly"


@pytest.mark.unit
@pytest.mark.phase3
class TestWSIDataset:
    """Tests for WSI dataset loader"""

    def test_dataset_initialization(self, sample_wsi_manifest, temp_dir):
        """Test WSI dataset can be initialized"""
        from mil_model.dataset import WSIDataset

        # Save manifest
        manifest_path = temp_dir / "manifest.csv"
        sample_wsi_manifest.to_csv(manifest_path, index=False)

        dataset = WSIDataset(
            manifest_path=manifest_path,
            features_dir=temp_dir / "features"
        )

        assert len(dataset) == len(sample_wsi_manifest)

    def test_dataset_getitem(self, sample_wsi_manifest, temp_dir):
        """Test dataset __getitem__ returns correct format"""
        from mil_model.dataset import WSIDataset

        # Save manifest
        manifest_path = temp_dir / "manifest.csv"
        sample_wsi_manifest.to_csv(manifest_path, index=False)

        # Create dummy features
        features_dir = temp_dir / "features"
        features_dir.mkdir()

        patient_id = sample_wsi_manifest.iloc[0]['patient_id']
        features = torch.randn(100, 1024)
        torch.save(features, features_dir / f"{patient_id}_features.pt")

        # Load dataset
        dataset = WSIDataset(manifest_path=manifest_path, features_dir=features_dir)

        # Get item
        features, label = dataset[0]

        assert features.shape == (100, 1024)
        assert label in [0, 1]

    def test_dataset_class_balance(self, sample_wsi_manifest, temp_dir):
        """Test dataset reports class distribution"""
        from mil_model.dataset import WSIDataset

        manifest_path = temp_dir / "manifest.csv"
        sample_wsi_manifest.to_csv(manifest_path, index=False)

        dataset = WSIDataset(manifest_path=manifest_path, features_dir=temp_dir)

        # Get class distribution
        labels = [sample_wsi_manifest.iloc[i]['pCR'] for i in range(len(sample_wsi_manifest))]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos

        assert n_pos > 0 and n_neg > 0, "Dataset should have both classes"


@pytest.mark.unit
@pytest.mark.phase3
class TestTrainingLoop:
    """Tests for training loop components"""

    def test_train_epoch_runs(self, trained_mil_model):
        """Test that training epoch runs without errors"""
        from mil_model.train import train_epoch
        from torch.utils.data import DataLoader, TensorDataset

        # Create dummy dataset
        features = [torch.randn(100, 1024) for _ in range(10)]
        labels = torch.randint(0, 2, (10,))
        dataset = list(zip(features, labels))

        # Use a simple collate function
        def collate_fn(batch):
            return batch[0]  # Return first item (single WSI per batch)

        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

        optimizer = torch.optim.Adam(trained_mil_model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Run one epoch
        avg_loss, accuracy = train_epoch(
            model=trained_mil_model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device='cpu'
        )

        assert isinstance(avg_loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1

    def test_validation_loop_runs(self, trained_mil_model):
        """Test that validation loop runs without errors"""
        from mil_model.train import validate
        from torch.utils.data import DataLoader

        # Create dummy dataset
        features = [torch.randn(100, 1024) for _ in range(10)]
        labels = torch.randint(0, 2, (10,))
        dataset = list(zip(features, labels))

        def collate_fn(batch):
            return batch[0]

        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

        criterion = nn.CrossEntropyLoss()

        # Run validation
        avg_loss, auc = validate(
            model=trained_mil_model,
            dataloader=dataloader,
            criterion=criterion,
            device='cpu'
        )

        assert isinstance(avg_loss, float)
        assert isinstance(auc, float)
        assert 0 <= auc <= 1

    def test_checkpoint_save_load(self, trained_mil_model, temp_dir):
        """Test saving and loading training checkpoints"""
        from mil_model.train import save_checkpoint, load_checkpoint

        checkpoint_path = temp_dir / "checkpoint.pth"

        # Save checkpoint
        save_checkpoint(
            model=trained_mil_model,
            optimizer=torch.optim.Adam(trained_mil_model.parameters()),
            epoch=10,
            val_auc=0.85,
            path=checkpoint_path
        )

        assert checkpoint_path.exists()

        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path)

        assert checkpoint['epoch'] == 10
        assert checkpoint['val_auc'] == 0.85
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint


@pytest.mark.unit
@pytest.mark.phase3
class TestAttentionHeatmap:
    """Tests for attention heatmap generation"""

    def test_create_attention_heatmap(self, sample_attention_weights, sample_patch_coords):
        """Test creation of attention heatmap from weights"""
        from mil_model.attention_mil import create_spatial_heatmap

        slide_dims = (5120, 5120)  # 10x10 grid of 512x512 patches

        heatmap = create_spatial_heatmap(
            attention_weights=sample_attention_weights.numpy(),
            patch_coords=sample_patch_coords,
            slide_dims=slide_dims
        )

        assert heatmap.shape == slide_dims
        assert heatmap.min() >= 0
        assert heatmap.max() <= 1

    def test_heatmap_highlights_top_patches(self):
        """Test that heatmap highlights top-attended regions"""
        from mil_model.attention_mil import create_spatial_heatmap

        # Create attention with one very high weight
        attention = np.zeros(100)
        attention[50] = 0.9  # High attention on patch 50
        attention = attention / attention.sum()

        coords = [(i % 10) * 512, (i // 10) * 512] for i in range(100)]

        heatmap = create_spatial_heatmap(attention, coords, (5120, 5120))

        # Find max location
        max_y, max_x = np.unravel_index(heatmap.argmax(), heatmap.shape)

        # Should be near patch 50's location (x=0, y=2560)
        expected_y = 2560  # 5 * 512
        assert abs(max_y - expected_y) < 512, "Max not at expected location"


@pytest.mark.unit
@pytest.mark.phase3
class TestMetrics:
    """Tests for evaluation metrics"""

    def test_auc_calculation(self):
        """Test AUC calculation"""
        from mil_model.metrics import calculate_auc

        # Perfect predictions
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])

        auc = calculate_auc(y_true, y_pred)

        assert auc == 1.0, "Perfect predictions should have AUC=1.0"

    def test_balanced_accuracy(self):
        """Test balanced accuracy calculation"""
        from mil_model.metrics import calculate_balanced_accuracy

        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])

        balanced_acc = calculate_balanced_accuracy(y_true, y_pred)

        # Sensitivity = 2/3, Specificity = 2/3, Balanced = 2/3
        assert abs(balanced_acc - 0.667) < 0.01

    def test_calibration_error(self):
        """Test calibration error calculation"""
        from mil_model.metrics import calculate_calibration_error

        y_true = np.array([0, 0, 1, 1])
        y_pred_prob = np.array([0.1, 0.2, 0.8, 0.9])

        calibration_error = calculate_calibration_error(y_true, y_pred_prob)

        assert calibration_error >= 0, "Calibration error should be non-negative"
        assert calibration_error <= 1, "Calibration error should be <= 1"


@pytest.mark.unit
@pytest.mark.phase3
class TestErrorHandling:
    """Tests for error handling"""

    def test_empty_bag_raises_error(self):
        """Test that empty feature bag raises appropriate error"""
        from mil_model.attention_mil import AttentionMIL

        model = AttentionMIL(feature_dim=1024, hidden_dim=256, n_classes=2)

        empty_features = torch.empty(0, 1024)

        with pytest.raises((ValueError, RuntimeError)):
            model(empty_features)

    def test_mismatched_feature_dim_raises_error(self):
        """Test that mismatched feature dimensions raise error"""
        from mil_model.attention_mil import AttentionMIL

        model = AttentionMIL(feature_dim=1024, hidden_dim=256, n_classes=2)

        wrong_dim_features = torch.randn(100, 512)  # 512 instead of 1024

        with pytest.raises((ValueError, RuntimeError)):
            model(wrong_dim_features)

    def test_invalid_label_raises_error(self):
        """Test that invalid labels raise error"""
        from mil_model.train import train_epoch

        model = torch.nn.Linear(1024, 2)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        # Invalid label (should be 0 or 1, not 5)
        features = [torch.randn(100, 1024)]
        labels = torch.tensor([5])

        dataset = list(zip(features, labels))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1,
            collate_fn=lambda x: x[0]
        )

        with pytest.raises((ValueError, RuntimeError)):
            train_epoch(model, dataloader, optimizer, criterion, 'cpu')
