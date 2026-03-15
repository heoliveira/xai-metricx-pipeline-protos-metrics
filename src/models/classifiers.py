"""Classifiers used in METRICX experiments.

Supported classifiers
---------------------
- Random Forest  (``RF``)
- XGBoost        (``XGB``)
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Default hyper-parameters chosen for reproducible experiments
_RF_DEFAULTS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": 42,
    "n_jobs": -1,
}

_XGB_DEFAULTS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

SUPPORTED_CLASSIFIERS = ("RF", "XGB")


def build_classifier(
    name: str,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Instantiate a classifier by name.

    Parameters
    ----------
    name:
        Classifier identifier: ``'RF'`` or ``'XGB'``.
    params:
        Hyper-parameter overrides.  Merged on top of the defaults.

    Returns
    -------
    Unfitted scikit-learn–compatible classifier.
    """
    name = name.upper()
    if name == "RF":
        cfg = {**_RF_DEFAULTS, **(params or {})}
        return RandomForestClassifier(**cfg)
    if name == "XGB":
        cfg = {**_XGB_DEFAULTS, **(params or {})}
        # Remove deprecated kwarg if present (XGBoost >= 2.0 removed it)
        cfg.pop("use_label_encoder", None)
        return XGBClassifier(**cfg)
    raise ValueError(
        f"Unknown classifier '{name}'. "
        f"Supported: {SUPPORTED_CLASSIFIERS}"
    )


def train_classifier(
    clf: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """Fit a classifier and return it.

    Parameters
    ----------
    clf:
        Unfitted scikit-learn–compatible classifier.
    X_train:
        Training feature matrix.
    y_train:
        Training labels.

    Returns
    -------
    Fitted classifier.
    """
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(
    clf: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Compute classification performance metrics.

    Parameters
    ----------
    clf:
        Fitted classifier.
    X_test:
        Test feature matrix.
    y_test:
        True test labels.

    Returns
    -------
    Dictionary with ``accuracy``, ``f1`` and ``roc_auc`` scores.
    """
    y_pred = clf.predict(X_test)
    y_prob = (
        clf.predict_proba(X_test)[:, 1]
        if hasattr(clf, "predict_proba")
        else None
    )

    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        except ValueError:
            metrics["roc_auc"] = float("nan")

    logger.info(
        "Classifier performance – acc=%.4f  f1=%.4f  auc=%.4f",
        metrics["accuracy"],
        metrics["f1"],
        metrics.get("roc_auc", float("nan")),
    )
    return metrics
