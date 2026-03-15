"""Tests for classifier building, training and evaluation."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from models.classifiers import (
    build_classifier,
    evaluate_classifier,
    train_classifier,
)


@pytest.fixture
def binary_dataset():
    """Small synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    split = 160
    return X[:split], X[split:], y[:split], y[split:]


class TestBuildClassifier:
    def test_rf_returns_random_forest(self):
        from sklearn.ensemble import RandomForestClassifier

        clf = build_classifier("RF")
        assert isinstance(clf, RandomForestClassifier)

    def test_xgb_returns_xgboost(self):
        from xgboost import XGBClassifier

        clf = build_classifier("XGB")
        assert isinstance(clf, XGBClassifier)

    def test_case_insensitive(self):
        clf = build_classifier("rf")
        from sklearn.ensemble import RandomForestClassifier

        assert isinstance(clf, RandomForestClassifier)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown classifier"):
            build_classifier("SVM")

    def test_custom_params_applied(self):
        clf = build_classifier("RF", params={"n_estimators": 7})
        assert clf.n_estimators == 7


class TestTrainAndEvaluate:
    def test_rf_trains_and_evaluates(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        clf = build_classifier("RF", params={"n_estimators": 10})
        clf = train_classifier(clf, X_train, y_train)
        metrics = evaluate_classifier(clf, X_test, y_test)

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_xgb_trains_and_evaluates(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        clf = build_classifier("XGB", params={"n_estimators": 10})
        clf = train_classifier(clf, X_train, y_train)
        metrics = evaluate_classifier(clf, X_test, y_test)

        assert metrics["accuracy"] > 0.5
