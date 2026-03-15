"""Tests for data loaders."""

import numpy as np
import pytest

from data.loaders import load_dataset


class TestLoadDataset:
    """Tests for :func:`data.loaders.load_dataset`."""

    def test_wdbc_fallback_loads_successfully(self):
        """WDBC dataset should load from sklearn built-in if no CSV exists."""
        X_train, X_test, y_train, y_test, feature_names = load_dataset(
            "WDBC", data_dir="/tmp/nonexistent_dir_metricx"
        )
        assert X_train.ndim == 2
        assert X_test.ndim == 2
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        assert len(feature_names) == X_train.shape[1]

    def test_wdbc_shapes_consistent(self):
        """Train/test shapes must match the expected split ratio."""
        X_train, X_test, y_train, y_test, _ = load_dataset(
            "WDBC", data_dir="/tmp/nonexistent_dir_metricx", test_size=0.2
        )
        total = len(y_train) + len(y_test)
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
        # Allow ±1 sample due to rounding
        assert abs(len(y_test) / total - 0.2) < 0.02

    def test_wdbc_scaled_features(self):
        """Features should be scaled (roughly zero mean)."""
        X_train, _, _, _, _ = load_dataset(
            "WDBC", data_dir="/tmp/nonexistent_dir_metricx", scale=True
        )
        # Mean of training features should be close to 0
        assert np.abs(X_train.mean()) < 0.1

    def test_unknown_dataset_raises(self):
        """An unknown dataset name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("UNKNOWN_DS")

    def test_missing_csv_raises(self):
        """A known dataset with a missing CSV should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset("FHS", data_dir="/tmp/nonexistent_dir_metricx")
