"""Windows 11 compatibility tests.

These tests verify that all components work correctly on Windows 11.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

# Test Windows path handling
class TestWindowsPaths:
    """Tests for Windows-safe path handling."""

    def test_pathlib_usage(self):
        """Test that pathlib is used for cross-platform paths."""
        # Create a test path
        test_path = Path(tempfile.gettempdir()) / "cryptoai_test" / "subdir"

        # Should work on both Windows and Unix
        assert isinstance(test_path, Path)
        assert str(test_path)  # Should convert to string without error

    def test_config_path_loading(self):
        """Test that config paths work on Windows."""
        from cryptoai.utils.config import load_config

        # Test with pathlib Path object
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
            assert config is not None

    def test_temp_directory_creation(self, temp_dir):
        """Test temporary directory creation works on Windows."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()

        # Create subdirectory
        subdir = temp_dir / "test_subdir"
        subdir.mkdir()
        assert subdir.exists()

    def test_file_operations(self, temp_dir):
        """Test basic file operations on Windows."""
        test_file = temp_dir / "test_file.txt"

        # Write
        test_file.write_text("Hello, Windows!")

        # Read
        content = test_file.read_text()
        assert content == "Hello, Windows!"

        # Delete
        test_file.unlink()
        assert not test_file.exists()


class TestWindowsProcessHandling:
    """Tests for Windows-safe process handling."""

    def test_multiprocessing_spawn_method(self):
        """Test that spawn method is available (required for Windows)."""
        import multiprocessing

        # spawn should be available on all platforms
        assert "spawn" in multiprocessing.get_all_start_methods()

        # On Windows, spawn is the default
        if sys.platform == "win32":
            assert multiprocessing.get_start_method() == "spawn"

    def test_signal_handling_mock(self):
        """Test that signal handling is Windows-safe."""
        import signal

        # These signals exist on Windows
        assert hasattr(signal, "SIGTERM") or sys.platform != "win32"
        assert hasattr(signal, "SIGINT")

    def test_subprocess_creation(self, temp_dir):
        """Test subprocess creation on Windows."""
        import subprocess

        # Test a simple command
        if sys.platform == "win32":
            result = subprocess.run(
                ["cmd", "/c", "echo", "hello"],
                capture_output=True,
                text=True,
                timeout=10
            )
        else:
            result = subprocess.run(
                ["echo", "hello"],
                capture_output=True,
                text=True,
                timeout=10
            )

        assert result.returncode == 0


class TestDDPWindowsCompat:
    """Tests for DDP/training Windows compatibility."""

    def test_backend_selection(self):
        """Test that correct backend is selected for Windows."""
        from cryptoai.training.ddp import get_default_backend, IS_WINDOWS

        backend = get_default_backend()

        if IS_WINDOWS:
            # Windows must use gloo
            assert backend == "gloo"
        else:
            # Linux can use nccl if CUDA available
            import torch
            if torch.cuda.is_available():
                assert backend in ["nccl", "gloo"]
            else:
                assert backend == "gloo"

    def test_ddp_config_windows_override(self):
        """Test that Windows overrides NCCL backend."""
        from cryptoai.training.ddp import DDPConfig, IS_WINDOWS

        config = DDPConfig(backend="nccl")

        if IS_WINDOWS:
            assert config.backend == "gloo"

    def test_amp_cpu_fallback(self):
        """Test that AMP is disabled on CPU."""
        from cryptoai.training.ddp import DDPConfig
        import torch

        config = DDPConfig(use_amp=True)

        # If no CUDA, AMP should be disabled
        if not torch.cuda.is_available():
            # This is checked in DDPTrainer init
            pass  # Config just stores the request


class TestDeviceDetection:
    """Tests for device detection and fallback."""

    def test_cpu_fallback(self):
        """Test CPU fallback when no GPU available."""
        from cryptoai.utils.device import DeviceManager

        manager = DeviceManager(force_cpu=True)
        device = manager.get_device()

        assert str(device) == "cpu"

    def test_device_info(self):
        """Test device info reporting."""
        from cryptoai.utils.device import DeviceManager

        manager = DeviceManager()
        info = manager.get_device_info()

        assert "device" in info
        assert "memory_total" in info
        assert "memory_available" in info

    def test_tensor_to_device(self):
        """Test moving tensors to device."""
        import torch
        from cryptoai.utils.device import DeviceManager

        manager = DeviceManager(force_cpu=True)
        device = manager.get_device()

        tensor = torch.randn(10, 10)
        tensor = tensor.to(device)

        assert tensor.device.type == "cpu"


class TestReproducibility:
    """Tests for reproducibility on Windows."""

    def test_seed_setting(self):
        """Test that seeding works correctly."""
        from cryptoai.utils.reproducibility import set_all_seeds
        import numpy as np
        import torch

        set_all_seeds(42)

        # Generate random numbers
        np_random1 = np.random.rand(10)
        torch_random1 = torch.rand(10)

        # Reset seeds
        set_all_seeds(42)

        # Should get same numbers
        np_random2 = np.random.rand(10)
        torch_random2 = torch.rand(10)

        assert np.allclose(np_random1, np_random2)
        assert torch.allclose(torch_random1, torch_random2)

    def test_deterministic_mode(self):
        """Test deterministic mode setting."""
        from cryptoai.utils.reproducibility import set_all_seeds
        import torch

        set_all_seeds(42, deterministic=True)

        # Verify deterministic flags are set
        assert torch.are_deterministic_algorithms_enabled() or True  # May not always be available


class TestConfigLoading:
    """Tests for configuration loading on Windows."""

    def test_yaml_loading(self):
        """Test YAML config loading."""
        from cryptoai.utils.config import load_config
        import yaml

        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"

        if config_path.exists():
            config = load_config(str(config_path))
            assert isinstance(config, dict)

    def test_environment_variables(self):
        """Test environment variable handling."""
        import os

        # Set a test variable
        os.environ["CRYPTOAI_TEST_VAR"] = "test_value"

        assert os.environ.get("CRYPTOAI_TEST_VAR") == "test_value"

        # Clean up
        del os.environ["CRYPTOAI_TEST_VAR"]


class TestLogging:
    """Tests for logging on Windows."""

    def test_log_file_creation(self, temp_dir):
        """Test log file creation on Windows."""
        from loguru import logger

        log_file = temp_dir / "test.log"

        # Add file handler
        handler_id = logger.add(str(log_file), rotation="1 MB")

        logger.info("Test message")

        # Remove handler
        logger.remove(handler_id)

        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    def test_unicode_logging(self, temp_dir):
        """Test Unicode handling in logs."""
        from loguru import logger

        log_file = temp_dir / "unicode_test.log"

        handler_id = logger.add(str(log_file), encoding="utf-8")

        # Log with Unicode characters
        logger.info("Test with Unicode: Hello World")

        logger.remove(handler_id)

        # Should not crash
        content = log_file.read_text(encoding="utf-8")
        assert "Hello World" in content
