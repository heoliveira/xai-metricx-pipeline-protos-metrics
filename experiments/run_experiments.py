"""METRICX experiment runner.

Usage
-----
Run all datasets::

    python experiments/run_experiments.py

Run a specific dataset with specific classifiers and XAI methods::

    python experiments/run_experiments.py \\
        --datasets WDBC \\
        --classifiers RF XGB \\
        --xai SHAP LIME \\
        --output results/metricx_results.csv
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running directly from the experiments/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from pipeline.pipeline import run_all_datasets, run_pipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("metricx.runner")

ALL_DATASETS = ["FHS", "SEPSIS", "STROKE", "UCI-THYROID-DXBIN", "WDBC"]
ALL_CLASSIFIERS = ["RF", "XGB"]
ALL_XAI = ["SHAP", "LIME"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="METRICX – XAI evaluation pipeline runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Datasets to process.",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=ALL_CLASSIFIERS,
        choices=ALL_CLASSIFIERS,
        help="Classifiers to train.",
    )
    parser.add_argument(
        "--xai",
        nargs="+",
        default=ALL_XAI,
        choices=ALL_XAI,
        dest="xai_methods",
        help="XAI methods to evaluate.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing dataset CSV files.",
    )
    parser.add_argument(
        "--output",
        default="results/metricx_results.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top features for faithfulness metrics.",
    )
    parser.add_argument(
        "--n-explain",
        type=int,
        default=100,
        help="Number of test samples to explain per experiment.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("METRICX experiment runner")
    logger.info("  Datasets    : %s", args.datasets)
    logger.info("  Classifiers : %s", args.classifiers)
    logger.info("  XAI methods : %s", args.xai_methods)

    results = run_all_datasets(
        datasets=args.datasets,
        classifier_names=args.classifiers,
        xai_methods=args.xai_methods,
        data_dir=args.data_dir,
        random_state=args.random_state,
        top_k=args.top_k,
        n_explain_samples=args.n_explain,
    )

    if results.empty:
        logger.warning("No results produced. Check dataset files.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)

    # Pretty-print summary
    print("\n=== METRICX Results Summary ===")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
