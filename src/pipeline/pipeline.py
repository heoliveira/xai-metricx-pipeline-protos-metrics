"""METRICX experiment pipeline.

Orchestrates the full end-to-end workflow:

1. Load dataset
2. Train classifiers (RF, XGB)
3. Evaluate classifiers
4. Generate XAI explanations (SHAP, LIME)
5. Compute XAI evaluation metrics
6. Return / persist results
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.loaders import load_dataset
from metrics.xai_metrics import compute_all_metrics
from models.classifiers import (
    build_classifier,
    evaluate_classifier,
    train_classifier,
)
from xai.explainers import Explanation, explain_lime, explain_shap

logger = logging.getLogger(__name__)


def run_pipeline(
    dataset_name: str,
    classifier_names: Optional[List[str]] = None,
    xai_methods: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    top_k: int = 5,
    n_explain_samples: int = 100,
    classifier_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Execute the full METRICX pipeline for a single dataset.

    Parameters
    ----------
    dataset_name:
        One of ``FHS``, ``SEPSIS``, ``STROKE``, ``UCI-THYROID-DXBIN``,
        ``WDBC``.
    classifier_names:
        List of classifier identifiers to use.  Defaults to
        ``['RF', 'XGB']``.
    xai_methods:
        List of XAI method identifiers to use.  Defaults to
        ``['SHAP', 'LIME']``.
    data_dir:
        Directory containing the dataset CSV files.
    test_size:
        Fraction of data used for testing.
    random_state:
        Random seed for reproducibility.
    top_k:
        Number of top features for faithfulness metrics.
    n_explain_samples:
        Number of test samples to explain (controls runtime).
    classifier_params:
        Optional mapping ``{clf_name: {param: value, ...}}`` to override
        default hyper-parameters.

    Returns
    -------
    :class:`pandas.DataFrame`
        One row per ``(dataset, classifier, xai_method)`` combination
        with columns for classifier performance and XAI metrics.
    """
    classifier_names = classifier_names or ["RF", "XGB"]
    xai_methods = xai_methods or ["SHAP", "LIME"]
    classifier_params = classifier_params or {}

    logger.info("=== METRICX pipeline: dataset=%s ===", dataset_name)

    X_train, X_test, y_train, y_test, feature_names = load_dataset(
        name=dataset_name,
        data_dir=data_dir,
        test_size=test_size,
        random_state=random_state,
    )

    rows = []

    for clf_name in classifier_names:
        logger.info("  Training classifier: %s", clf_name)
        clf = build_classifier(clf_name, classifier_params.get(clf_name))
        clf = train_classifier(clf, X_train, y_train)
        clf_metrics = evaluate_classifier(clf, X_test, y_test)

        for xai_name in xai_methods:
            logger.info("    Generating explanations: %s", xai_name)

            try:
                explanation = _generate_explanation(
                    xai_name=xai_name,
                    clf=clf,
                    X_train=X_train,
                    X_test=X_test,
                    feature_names=feature_names,
                    n_explain_samples=n_explain_samples,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "    Failed to generate %s explanation: %s",
                    xai_name,
                    exc,
                )
                continue

            xai_scores = compute_all_metrics(
                clf=clf,
                X=X_test[:n_explain_samples],
                explanation=explanation,
                top_k=top_k,
                n_samples=n_explain_samples,
            )

            row: Dict[str, Any] = {
                "dataset": dataset_name,
                "classifier": clf_name,
                "xai_method": xai_name,
                **clf_metrics,
                **xai_scores,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def run_all_datasets(
    datasets: Optional[List[str]] = None,
    **pipeline_kwargs: Any,
) -> pd.DataFrame:
    """Run the pipeline over multiple datasets and concatenate results.

    Parameters
    ----------
    datasets:
        List of dataset names.  Defaults to all five supported datasets.
    **pipeline_kwargs:
        Keyword arguments forwarded to :func:`run_pipeline`.

    Returns
    -------
    :class:`pandas.DataFrame`
        Concatenated results for all datasets.
    """
    if datasets is None:
        datasets = ["FHS", "SEPSIS", "STROKE", "UCI-THYROID-DXBIN", "WDBC"]

    frames = []
    for ds in datasets:
        try:
            df = run_pipeline(ds, **pipeline_kwargs)
            frames.append(df)
        except FileNotFoundError as exc:
            logger.warning("Skipping dataset '%s': %s", ds, exc)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_explanation(
    xai_name: str,
    clf: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    n_explain_samples: int,
) -> Explanation:
    """Dispatch to the appropriate explainer function."""
    xai_name = xai_name.upper()
    if xai_name == "SHAP":
        return explain_shap(
            clf,
            X_test,
            feature_names=feature_names,
            max_samples=n_explain_samples,
        )
    if xai_name == "LIME":
        return explain_lime(
            clf,
            X_train,
            X_test,
            feature_names=feature_names,
            max_samples=n_explain_samples,
        )
    raise ValueError(
        f"Unknown XAI method '{xai_name}'. Supported: SHAP, LIME"
    )
