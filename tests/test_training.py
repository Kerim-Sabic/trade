"""Unit tests for training module.

Windows 11 Compatible - Uses gloo backend and CPU.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from cryptoai.training.ddp import (
    DDPConfig,
    DDPTrainer,
    get_default_backend,
    IS_WINDOWS,
)
from cryptoai.training.data_loader import (
    DataConfig,
    ExperienceDataset,
    create_synthetic_data,
)


class TestDDPConfig:
    """Tests for DDPConfig."""

    def test_default_backend_detection(self):
        """Test that backend is correctly detected for platform."""
        backend = get_default_backend()

        if IS_WINDOWS:
            assert backend == "gloo", "Windows should use gloo backend"
        else:
            # Linux with CUDA should use nccl, otherwise gloo
            if torch.cuda.is_available():
                assert backend == "nccl"
            else:
                assert backend == "gloo"

    def test_windows_forces_gloo(self):
        """Test that Windows config forces gloo backend."""
        config = DDPConfig(backend="nccl")

        if IS_WINDOWS:
            assert config.backend == "gloo", "Windows must use gloo"

    def test_world_size_detection(self):
        """Test that world size is auto-detected."""
        config = DDPConfig()

        assert config.world_size >= 1

    def test_checkpoint_dir_created(self, temp_dir):
        """Test that checkpoint directory is created."""
        checkpoint_dir = str(temp_dir / "checkpoints")
        config = DDPConfig(checkpoint_dir=checkpoint_dir)

        assert Path(checkpoint_dir).exists()


class TestDDPTrainer:
    """Tests for DDPTrainer base class."""

    def test_device_selection(self, device):
        """Test that correct device is selected."""
        model = nn.Linear(10, 10)
        config = DDPConfig(world_size=1)

        trainer = DDPTrainer(model, config, rank=0)

        # On CPU-only systems, should use CPU
        if not torch.cuda.is_available():
            assert trainer.device == torch.device("cpu")
            assert not trainer.is_cuda

    def test_amp_disabled_on_cpu(self, device):
        """Test that AMP is disabled on CPU."""
        model = nn.Linear(10, 10)
        config = DDPConfig(world_size=1, use_amp=True)

        trainer = DDPTrainer(model, config, rank=0)

        if not torch.cuda.is_available():
            assert not trainer.use_amp, "AMP should be disabled on CPU"

    def test_model_to_device(self, device):
        """Test that model is moved to correct device."""
        model = nn.Linear(10, 10)
        config = DDPConfig(world_size=1)

        trainer = DDPTrainer(model, config, rank=0)

        # Check model is on trainer's device
        for param in trainer.model.parameters():
            assert param.device == trainer.device


class TestExperienceDataset:
    """Tests for ExperienceDataset."""

    def test_initialization(self):
        """Test dataset initialization."""
        dataset = ExperienceDataset(
            capacity=1000,
            state_dim=100,
            action_dim=4,
            sequence_length=50,
        )

        assert len(dataset) == 0
        assert dataset.capacity == 1000

    def test_add_experience(self):
        """Test adding experience to dataset."""
        dataset = ExperienceDataset(
            capacity=100,
            state_dim=100,
            action_dim=4,
            sequence_length=50,
        )

        state = np.random.randn(50, 100).astype(np.float32)
        action = np.random.randn(4).astype(np.float32)
        next_state = np.random.randn(50, 100).astype(np.float32)

        dataset.add(state, action, reward=1.0, next_state=next_state, done=False)

        assert len(dataset) == 1

    def test_getitem(self):
        """Test retrieving items from dataset."""
        dataset = ExperienceDataset(
            capacity=100,
            state_dim=100,
            action_dim=4,
            sequence_length=50,
        )

        state = np.random.randn(50, 100).astype(np.float32)
        action = np.random.randn(4).astype(np.float32)
        next_state = np.random.randn(50, 100).astype(np.float32)

        dataset.add(state, action, reward=1.0, next_state=next_state, done=False)

        item = dataset[0]

        assert "state" in item
        assert "action" in item
        assert "reward" in item
        assert "next_state" in item
        assert "done" in item

    def test_circular_buffer(self):
        """Test that dataset acts as circular buffer."""
        capacity = 10
        dataset = ExperienceDataset(
            capacity=capacity,
            state_dim=100,
            action_dim=4,
            sequence_length=50,
        )

        # Add more than capacity
        for i in range(capacity * 2):
            state = np.random.randn(50, 100).astype(np.float32)
            action = np.random.randn(4).astype(np.float32)
            next_state = np.random.randn(50, 100).astype(np.float32)
            dataset.add(state, action, reward=float(i), next_state=next_state, done=False)

        assert len(dataset) == capacity

    def test_sample_batch(self):
        """Test batch sampling."""
        dataset = ExperienceDataset(
            capacity=100,
            state_dim=100,
            action_dim=4,
            sequence_length=50,
        )

        # Add experiences
        for i in range(50):
            state = np.random.randn(50, 100).astype(np.float32)
            action = np.random.randn(4).astype(np.float32)
            next_state = np.random.randn(50, 100).astype(np.float32)
            dataset.add(state, action, reward=float(i), next_state=next_state, done=False)

        batch = dataset.sample_batch(batch_size=16)

        assert batch["state"].shape[0] == 16
        assert batch["action"].shape[0] == 16


class TestSyntheticDataCreation:
    """Tests for synthetic data creation."""

    def test_create_synthetic_data(self, temp_dir):
        """Test synthetic data creation."""
        output_path = str(temp_dir / "test_data.h5")

        create_synthetic_data(
            output_path=output_path,
            n_samples=100,
            sequence_length=50,
            feature_dim=100,
            action_dim=4,
        )

        assert Path(output_path).exists()

    def test_synthetic_data_contents(self, temp_dir):
        """Test that synthetic data has correct contents."""
        import h5py

        output_path = str(temp_dir / "test_data.h5")
        n_samples = 100
        seq_len = 50
        feature_dim = 100
        action_dim = 4

        create_synthetic_data(
            output_path=output_path,
            n_samples=n_samples,
            sequence_length=seq_len,
            feature_dim=feature_dim,
            action_dim=action_dim,
        )

        with h5py.File(output_path, "r") as f:
            assert f["states"].shape == (n_samples, seq_len, feature_dim)
            assert f["actions"].shape == (n_samples, action_dim)
            assert f["rewards"].shape == (n_samples,)
            assert f["next_states"].shape == (n_samples, seq_len, feature_dim)
            assert f["dones"].shape == (n_samples,)
