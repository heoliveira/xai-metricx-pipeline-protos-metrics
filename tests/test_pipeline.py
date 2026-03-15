"""Integration test for the METRICX pipeline using the WDBC built-in dataset."""

import pytest

from pipeline.pipeline import run_pipeline


class TestPipelineIntegration:
    """End-to-end pipeline tests using the WDBC sklearn built-in."""

    def test_pipeline_returns_dataframe(self):
        """Pipeline should return a non-empty DataFrame."""
        df = run_pipeline(
            dataset_name="WDBC",
            classifier_names=["RF"],
            xai_methods=["SHAP"],
            data_dir="/tmp/nonexistent_dir_metricx",
            n_explain_samples=30,
        )
        assert not df.empty

    def test_pipeline_columns_present(self):
        """Expected result columns should all be present."""
        df = run_pipeline(
            dataset_name="WDBC",
            classifier_names=["RF"],
            xai_methods=["SHAP"],
            data_dir="/tmp/nonexistent_dir_metricx",
            n_explain_samples=30,
        )
        expected_cols = {
            "dataset",
            "classifier",
            "xai_method",
            "accuracy",
            "f1",
            "roc_auc",
            "faithfulness_correlation",
            "comprehensiveness",
            "sufficiency",
            "avg_sensitivity",
            "complexity",
            "effective_complexity",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_pipeline_rf_xgb_shap(self):
        """Pipeline should produce two rows (one per classifier) with SHAP."""
        df = run_pipeline(
            dataset_name="WDBC",
            classifier_names=["RF", "XGB"],
            xai_methods=["SHAP"],
            data_dir="/tmp/nonexistent_dir_metricx",
            n_explain_samples=30,
        )
        assert len(df) == 2
        assert set(df["classifier"]) == {"RF", "XGB"}

    def test_pipeline_accuracy_reasonable(self):
        """Accuracy on WDBC should exceed a trivial baseline."""
        df = run_pipeline(
            dataset_name="WDBC",
            classifier_names=["RF"],
            xai_methods=["SHAP"],
            data_dir="/tmp/nonexistent_dir_metricx",
            n_explain_samples=30,
        )
        assert df.iloc[0]["accuracy"] > 0.7
