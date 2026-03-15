"""Data loaders for METRICX experiments.

Supported datasets
------------------
- FHS       : Framingham Heart Study
- SEPSIS    : Sepsis clinical dataset
- STROKE    : Stroke prediction dataset
- UCI-THYROID-DXBIN : UCI Thyroid disease (binary classification)
- WDBC      : Wisconsin Diagnostic Breast Cancer
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Mapping of dataset names to their expected CSV file names
DATASET_FILES = {
    "FHS": "fhs.csv",
    "SEPSIS": "sepsis.csv",
    "STROKE": "stroke.csv",
    "UCI-THYROID-DXBIN": "uci_thyroid_dxbin.csv",
    "WDBC": "wdbc.csv",
}

# Column that holds the target label for each dataset
TARGET_COLUMNS = {
    "FHS": "TenYearCHD",
    "SEPSIS": "SepsisLabel",
    "STROKE": "stroke",
    "UCI-THYROID-DXBIN": "diagnosis",
    "WDBC": "diagnosis",
}


def _encode_and_clean(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop rows with missing values and encode categoricals.

    Parameters
    ----------
    df:
        Raw dataframe loaded from disk.
    target_col:
        Name of the target column.

    Returns
    -------
    X:
        Feature matrix with numeric dtypes.
    y:
        Binary target series (0 / 1).
    """
    df = df.dropna().reset_index(drop=True)

    # Separate target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Encode object columns
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Encode target if needed
    if y.dtype == object or str(y.dtype) == "category":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)
    else:
        y = y.astype(int)

    return X.astype(float), y


def load_dataset(
    name: str,
    data_dir: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Load and split a dataset by name.

    For the WDBC dataset the scikit-learn built-in version is used as a
    fallback when no CSV file is available, making experiments runnable
    without external data files.  For the remaining datasets a CSV file
    must be present in *data_dir*.

    Parameters
    ----------
    name:
        Dataset identifier.  Must be one of: ``FHS``, ``SEPSIS``,
        ``STROKE``, ``UCI-THYROID-DXBIN``, ``WDBC``.
    data_dir:
        Directory that contains the CSV files.  Defaults to the
        ``data/`` folder at the project root.
    test_size:
        Fraction of samples reserved for testing.
    random_state:
        Random seed for reproducibility.
    scale:
        Whether to apply :class:`~sklearn.preprocessing.StandardScaler`
        to the feature matrices.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    name = name.upper()
    if name not in DATASET_FILES:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Supported datasets: {list(DATASET_FILES)}"
        )

    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[3] / "data"
    else:
        data_dir = Path(data_dir)

    csv_path = data_dir / DATASET_FILES[name]

    # WDBC: prefer CSV but fall back to sklearn built-in
    if name == "WDBC" and not csv_path.exists():
        logger.info("WDBC CSV not found – using sklearn built-in dataset.")
        bunch = load_breast_cancer()
        X_raw = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        y_raw = pd.Series(bunch.target, name="diagnosis")
        feature_names = list(bunch.feature_names)
    else:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {csv_path}\n"
                "Place the CSV file in the data/ directory and re-run."
            )
        df = pd.read_csv(csv_path)
        target_col = TARGET_COLUMNS[name]
        X_raw, y_raw = _encode_and_clean(df, target_col)
        feature_names = list(X_raw.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw if isinstance(X_raw, np.ndarray) else X_raw.values,
        y_raw.values,
        test_size=test_size,
        random_state=random_state,
        stratify=y_raw.values,
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    logger.info(
        "Loaded dataset '%s': %d train / %d test samples, %d features.",
        name,
        len(X_train),
        len(X_test),
        len(feature_names),
    )

    return X_train, X_test, y_train, y_test, feature_names
