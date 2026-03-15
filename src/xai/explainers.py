"""XAI explainers used in METRICX experiments.

Supported methods
-----------------
- SHAP  (SHapley Additive exPlanations) via ``shap`` library
- LIME  (Local Interpretable Model-agnostic Explanations) via ``lime``

Both return a unified :class:`Explanation` object that stores a 2-D
importance matrix of shape ``(n_samples, n_features)``.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Container for XAI explanation results.

    Attributes
    ----------
    method:
        Name of the XAI method used (e.g. ``'SHAP'``, ``'LIME'``).
    importances:
        2-D array of shape ``(n_samples, n_features)`` with per-sample
        feature importances.
    feature_names:
        Ordered list of feature names.
    """

    method: str
    importances: np.ndarray
    feature_names: List[str] = field(default_factory=list)

    @property
    def mean_importances(self) -> np.ndarray:
        """Mean absolute importance per feature across all samples."""
        return np.mean(np.abs(self.importances), axis=0)


def explain_shap(
    clf: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 200,
) -> Explanation:
    """Compute SHAP values for a fitted classifier.

    Uses :class:`shap.TreeExplainer` for tree-based models (Random
    Forest, XGBoost) and :class:`shap.KernelExplainer` as a fallback.

    Parameters
    ----------
    clf:
        Fitted classifier.
    X:
        Feature matrix to explain (typically the test set or a subset).
    feature_names:
        Feature names aligned with the columns of *X*.
    max_samples:
        Maximum number of samples to explain (for speed).

    Returns
    -------
    :class:`Explanation`
    """
    import shap  # noqa: PLC0415

    X_sub = X[:max_samples]

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_sub)
    except Exception:  # noqa: BLE001
        logger.warning(
            "TreeExplainer failed – falling back to KernelExplainer."
        )
        background = shap.sample(X_sub, min(50, len(X_sub)))
        explainer = shap.KernelExplainer(clf.predict_proba, background)
        shap_values = explainer.shap_values(X_sub, nsamples=50)

    # shap_values may be a list (one array per class) or a single array
    if isinstance(shap_values, list):
        # For binary classification take the positive class
        importances = shap_values[-1]
    else:
        importances = shap_values

    # Handle 3-D output (n_samples, n_features, n_outputs)
    if importances.ndim == 3:
        importances = importances[:, :, -1]

    return Explanation(
        method="SHAP",
        importances=importances,
        feature_names=feature_names or [f"f{i}" for i in range(X.shape[1])],
    )


def explain_lime(
    clf: Any,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 100,
    num_features: Optional[int] = None,
) -> Explanation:
    """Compute LIME explanations for a fitted classifier.

    Parameters
    ----------
    clf:
        Fitted classifier with a ``predict_proba`` method.
    X_train:
        Training data used to initialise the LIME explainer.
    X_explain:
        Samples to explain (typically the test set or a subset).
    feature_names:
        Feature names aligned with the columns.
    max_samples:
        Maximum number of samples to explain.
    num_features:
        Number of features included in each LIME explanation.  Defaults
        to ``min(10, n_features)``.

    Returns
    -------
    :class:`Explanation`
    """
    from lime.lime_tabular import LimeTabularExplainer  # noqa: PLC0415

    n_features = X_train.shape[1]
    names = feature_names or [f"f{i}" for i in range(n_features)]
    k = num_features or min(10, n_features)

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=names,
        mode="classification",
        discretize_continuous=True,
    )

    X_sub = X_explain[:max_samples]
    importances = np.zeros((len(X_sub), n_features))

    for i, sample in enumerate(X_sub):
        exp = explainer.explain_instance(
            sample,
            clf.predict_proba,
            num_features=k,
        )
        for feat_idx, feat_weight in exp.local_exp[1]:
                importances[i, feat_idx] = feat_weight

    return Explanation(
        method="LIME",
        importances=importances,
        feature_names=names,
    )
