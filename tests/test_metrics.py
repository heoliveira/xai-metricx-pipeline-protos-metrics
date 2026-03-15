"""Tests for XAI evaluation metrics."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from xai.explainers import Explanation, explain_shap
from metrics.xai_metrics import (
    avg_sensitivity,
    complexity,
    comprehensiveness,
    compute_all_metrics,
    effective_complexity,
    faithfulness_correlation,
    sufficiency,
)


@pytest.fixture(scope="module")
def clf_and_data():
    """Fitted RF classifier and small dataset."""
    X, y = make_classification(
        n_samples=300, n_features=10, n_informative=5, random_state=0
    )
    X_train, X_test = X[:240], X[240:]
    y_train = y[:240]
    clf = RandomForestClassifier(n_estimators=20, random_state=0)
    clf.fit(X_train, y_train)
    return clf, X_train, X_test


@pytest.fixture(scope="module")
def shap_explanation(clf_and_data):
    clf, X_train, X_test = clf_and_data
    return explain_shap(clf, X_test, max_samples=50)


class TestExplanation:
    def test_mean_importances_shape(self, shap_explanation):
        exp = shap_explanation
        assert exp.mean_importances.shape == (exp.importances.shape[1],)

    def test_importances_are_finite(self, shap_explanation):
        assert np.all(np.isfinite(shap_explanation.importances))


class TestFaithfulnessMetrics:
    def test_faithfulness_correlation_range(self, clf_and_data, shap_explanation):
        clf, _, X_test = clf_and_data
        score = faithfulness_correlation(clf, X_test, shap_explanation, n_samples=40)
        assert -1.0 <= score <= 1.0

    def test_comprehensiveness_is_float(self, clf_and_data, shap_explanation):
        clf, _, X_test = clf_and_data
        score = comprehensiveness(clf, X_test, shap_explanation, top_k=3, n_samples=40)
        assert isinstance(score, float)

    def test_sufficiency_is_float(self, clf_and_data, shap_explanation):
        clf, _, X_test = clf_and_data
        score = sufficiency(clf, X_test, shap_explanation, top_k=3, n_samples=40)
        assert isinstance(score, float)


class TestStabilityMetrics:
    def test_avg_sensitivity_non_negative(self, clf_and_data, shap_explanation):
        clf, _, X_test = clf_and_data
        score = avg_sensitivity(clf, X_test, shap_explanation, n_samples=20)
        assert score >= 0.0


class TestComplexityMetrics:
    def test_complexity_positive(self, shap_explanation):
        score = complexity(shap_explanation)
        assert score > 0.0

    def test_effective_complexity_range(self, shap_explanation):
        score = effective_complexity(shap_explanation)
        n_features = shap_explanation.importances.shape[1]
        assert 0.0 <= score <= n_features


class TestComputeAllMetrics:
    def test_all_keys_present(self, clf_and_data, shap_explanation):
        clf, _, X_test = clf_and_data
        results = compute_all_metrics(clf, X_test, shap_explanation, n_samples=30)
        expected_keys = {
            "faithfulness_correlation",
            "comprehensiveness",
            "sufficiency",
            "avg_sensitivity",
            "complexity",
            "effective_complexity",
        }
        assert set(results.keys()) == expected_keys

    def test_all_values_are_floats(self, clf_and_data, shap_explanation):
        clf, _, X_test = clf_and_data
        results = compute_all_metrics(clf, X_test, shap_explanation, n_samples=30)
        for k, v in results.items():
            assert isinstance(v, float), f"{k} is not a float"
