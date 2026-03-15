"""Quantitative metrics for evaluating XAI explanations.

This module implements the core METRICX evaluation metrics grouped into
three families:

Faithfulness
    How well the explanation reflects the model's actual decision
    process.

    * ``faithfulness_correlation``  – Pearson correlation between feature
      importances and the change in model prediction when a feature is
      individually perturbed (inspired by Samek et al., 2017).

    * ``sufficiency``               – Mean drop in predicted probability
      when only the top-k features are kept (lower = more sufficient).

    * ``comprehensiveness``         – Mean drop in predicted probability
      when the top-k features are removed (higher = more comprehensive).

Stability
    How robust explanations are to small input perturbations.

    * ``avg_sensitivity``           – Average max change in explanation
      norm when Gaussian noise is added to inputs (Yeh et al., 2019).

Complexity
    How concise the explanation is.

    * ``complexity``                – Average entropy of the normalised
      absolute importance distribution.

    * ``effective_complexity``      – Average number of features with
      absolute importance above a threshold.

References
----------
Samek, W. et al. (2017). Evaluating the visualization of what a deep
neural network has learned. *IEEE TNNLS*.

Yeh, C.-K. et al. (2019). On the (in)fidelity and sensitivity of
explanations. *NeurIPS*.

Doshi-Velez, F. & Kim, B. (2017). Towards a rigorous science of
interpretable machine learning. *arXiv:1702.08608*.
"""

import logging
from typing import Any, Optional

import numpy as np

from xai.explainers import Explanation

logger = logging.getLogger(__name__)


def _slice_importances(explanation: Explanation, n_samples: Optional[int]) -> np.ndarray:
    """Return the importance matrix optionally limited to *n_samples* rows."""
    if n_samples is None:
        return explanation.importances
    return explanation.importances[:n_samples]


# ---------------------------------------------------------------------------
# Faithfulness metrics
# ---------------------------------------------------------------------------


def faithfulness_correlation(
    clf: Any,
    X: np.ndarray,
    explanation: Explanation,
    n_samples: int = 100,
) -> float:
    """Pearson correlation between importance magnitudes and prediction change.

    For each sample the feature with the highest absolute importance is
    masked (set to its column mean) and the resulting prediction drop is
    recorded.  The final score is the Pearson correlation between the
    importance values and the prediction drops across all samples.

    A higher positive value indicates that the explanation is more
    faithful to the model.

    Parameters
    ----------
    clf:
        Fitted classifier.
    X:
        Feature matrix used for explanation (same rows as
        ``explanation.importances``).
    explanation:
        :class:`~src.xai.explainers.Explanation` object.
    n_samples:
        Maximum number of samples to use.

    Returns
    -------
    float
        Pearson correlation coefficient in ``[-1, 1]``.
    """
    X_sub = X[: min(n_samples, len(X))]
    imps = explanation.importances[: len(X_sub)]
    col_means = X_sub.mean(axis=0)

    base_probs = clf.predict_proba(X_sub)[:, 1]

    importance_vals = []
    prediction_drops = []

    for i in range(len(X_sub)):
        feat_idx = int(np.argmax(np.abs(imps[i])))
        imp_val = float(np.abs(imps[i, feat_idx]))

        x_masked = X_sub[i].copy()
        x_masked[feat_idx] = col_means[feat_idx]
        masked_prob = clf.predict_proba(x_masked.reshape(1, -1))[0, 1]

        importance_vals.append(imp_val)
        prediction_drops.append(float(base_probs[i] - masked_prob))

    corr = float(np.corrcoef(importance_vals, prediction_drops)[0, 1])
    if np.isnan(corr):
        corr = 0.0
    return corr


def comprehensiveness(
    clf: Any,
    X: np.ndarray,
    explanation: Explanation,
    top_k: int = 5,
    n_samples: int = 100,
) -> float:
    """Mean prediction drop when the top-k features are removed.

    A higher score means the top-k features identified by the explanation
    are more important to the model's predictions.

    Parameters
    ----------
    clf:
        Fitted classifier.
    X:
        Feature matrix aligned with ``explanation.importances``.
    explanation:
        :class:`~src.xai.explainers.Explanation` object.
    top_k:
        Number of top features to remove.
    n_samples:
        Maximum number of samples.

    Returns
    -------
    float
        Mean prediction drop (higher ≈ more comprehensive).
    """
    X_sub = X[: min(n_samples, len(X))]
    imps = explanation.importances[: len(X_sub)]
    col_means = X_sub.mean(axis=0)
    k = min(top_k, X_sub.shape[1])

    base_probs = clf.predict_proba(X_sub)[:, 1]
    drops = []

    for i in range(len(X_sub)):
        top_feats = np.argsort(np.abs(imps[i]))[-k:]
        x_masked = X_sub[i].copy()
        x_masked[top_feats] = col_means[top_feats]
        masked_prob = clf.predict_proba(x_masked.reshape(1, -1))[0, 1]
        drops.append(float(base_probs[i] - masked_prob))

    return float(np.mean(drops))


def sufficiency(
    clf: Any,
    X: np.ndarray,
    explanation: Explanation,
    top_k: int = 5,
    n_samples: int = 100,
) -> float:
    """Mean prediction drop when only the top-k features are kept.

    A lower (less negative) score indicates that the top-k features are
    sufficient to reproduce the original prediction.

    Parameters
    ----------
    clf:
        Fitted classifier.
    X:
        Feature matrix aligned with ``explanation.importances``.
    explanation:
        :class:`~src.xai.explainers.Explanation` object.
    top_k:
        Number of top features to keep.
    n_samples:
        Maximum number of samples.

    Returns
    -------
    float
        Mean prediction drop (closer to 0 ≈ more sufficient).
    """
    X_sub = X[: min(n_samples, len(X))]
    imps = explanation.importances[: len(X_sub)]
    col_means = X_sub.mean(axis=0)
    k = min(top_k, X_sub.shape[1])

    base_probs = clf.predict_proba(X_sub)[:, 1]
    drops = []

    for i in range(len(X_sub)):
        top_feats = np.argsort(np.abs(imps[i]))[-k:]
        x_kept = col_means.copy()
        x_kept[top_feats] = X_sub[i, top_feats]
        kept_prob = clf.predict_proba(x_kept.reshape(1, -1))[0, 1]
        drops.append(float(base_probs[i] - kept_prob))

    return float(np.mean(drops))


# ---------------------------------------------------------------------------
# Stability metrics
# ---------------------------------------------------------------------------


def avg_sensitivity(
    clf: Any,
    X: np.ndarray,
    explanation: Explanation,
    n_perturbations: int = 10,
    noise_std: float = 0.1,
    n_samples: int = 50,
) -> float:
    """Average max-sensitivity of the explanation under Gaussian noise.

    For each sample, *n_perturbations* noisy versions are generated.
    SHAP/LIME importances are approximated by computing the gradient of
    the model output w.r.t. the input perturbation (finite-difference
    proxy).  The final score is the mean over all samples of the maximum
    L2 norm of the importance change.

    Lower values indicate more stable (robust) explanations.

    Parameters
    ----------
    clf:
        Fitted classifier.
    X:
        Feature matrix aligned with ``explanation.importances``.
    explanation:
        :class:`~src.xai.explainers.Explanation` object.
    n_perturbations:
        Number of noise perturbations per sample.
    noise_std:
        Standard deviation of the Gaussian noise.
    n_samples:
        Maximum number of samples.

    Returns
    -------
    float
        Average max sensitivity (lower ≈ more stable).
    """
    X_sub = X[: min(n_samples, len(X))]
    imps = explanation.importances[: len(X_sub)]
    rng = np.random.default_rng(42)

    sensitivities = []

    for i in range(len(X_sub)):
        base_imp = imps[i]
        base_norm = np.linalg.norm(base_imp) + 1e-10
        max_diff = 0.0

        for _ in range(n_perturbations):
            noise = rng.normal(0, noise_std, size=X_sub[i].shape)
            x_noisy = X_sub[i] + noise
            # Proxy importance: gradient of prob w.r.t. feature (finite diff)
            prob_base = clf.predict_proba(X_sub[i].reshape(1, -1))[0, 1]
            prob_noisy = clf.predict_proba(x_noisy.reshape(1, -1))[0, 1]
            # Approximate perturbed importance as scaled noise direction
            delta_imp = noise * (prob_noisy - prob_base)
            diff = np.linalg.norm(base_imp - delta_imp) / base_norm
            if diff > max_diff:
                max_diff = diff

        sensitivities.append(max_diff)

    return float(np.mean(sensitivities))


# ---------------------------------------------------------------------------
# Complexity metrics
# ---------------------------------------------------------------------------


def complexity(
    explanation: Explanation,
    n_samples: Optional[int] = None,
) -> float:
    """Average entropy of the normalised importance distribution.

    Higher values indicate more complex (spread-out) explanations.

    Parameters
    ----------
    explanation:
        :class:`~src.xai.explainers.Explanation` object.
    n_samples:
        Maximum number of samples (``None`` = all).

    Returns
    -------
    float
        Mean entropy across samples (nats).
    """
    imps = _slice_importances(explanation, n_samples)
    abs_imps = np.abs(imps)
    row_sums = abs_imps.sum(axis=1, keepdims=True) + 1e-10
    probs = abs_imps / row_sums
    # Avoid log(0)
    probs = np.clip(probs, 1e-10, None)
    entropies = -np.sum(probs * np.log(probs), axis=1)
    return float(np.mean(entropies))


def effective_complexity(
    explanation: Explanation,
    threshold: float = 0.01,
    n_samples: Optional[int] = None,
) -> float:
    """Average number of features with absolute importance above a threshold.

    After normalising each sample's importance vector to sum to 1, counts
    how many features exceed *threshold*.

    Parameters
    ----------
    explanation:
        :class:`~src.xai.explainers.Explanation` object.
    threshold:
        Minimum normalised importance to count a feature as "active".
    n_samples:
        Maximum number of samples (``None`` = all).

    Returns
    -------
    float
        Mean effective complexity (lower = simpler explanation).
    """
    imps = _slice_importances(explanation, n_samples)
    abs_imps = np.abs(imps)
    row_sums = abs_imps.sum(axis=1, keepdims=True) + 1e-10
    norm_imps = abs_imps / row_sums
    counts = (norm_imps > threshold).sum(axis=1)
    return float(np.mean(counts))


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def compute_all_metrics(
    clf: Any,
    X: np.ndarray,
    explanation: Explanation,
    top_k: int = 5,
    n_samples: int = 100,
) -> dict:
    """Compute all METRICX metrics and return as a dictionary.

    Parameters
    ----------
    clf:
        Fitted classifier.
    X:
        Feature matrix aligned with ``explanation.importances``.
    explanation:
        :class:`~src.xai.explainers.Explanation` object.
    top_k:
        Number of features for faithfulness metrics.
    n_samples:
        Maximum number of samples for expensive metrics.

    Returns
    -------
    dict
        Keys: ``faithfulness_correlation``, ``comprehensiveness``,
        ``sufficiency``, ``avg_sensitivity``, ``complexity``,
        ``effective_complexity``.
    """
    results = {
        "faithfulness_correlation": faithfulness_correlation(
            clf, X, explanation, n_samples=n_samples
        ),
        "comprehensiveness": comprehensiveness(
            clf, X, explanation, top_k=top_k, n_samples=n_samples
        ),
        "sufficiency": sufficiency(
            clf, X, explanation, top_k=top_k, n_samples=n_samples
        ),
        "avg_sensitivity": avg_sensitivity(
            clf, X, explanation, n_samples=min(n_samples, 50)
        ),
        "complexity": complexity(explanation, n_samples=n_samples),
        "effective_complexity": effective_complexity(
            explanation, n_samples=n_samples
        ),
    }
    logger.info(
        "XAI metrics for %s – %s",
        explanation.method,
        {k: f"{v:.4f}" for k, v in results.items()},
    )
    return results
