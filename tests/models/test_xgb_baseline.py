import pytest
import pandas as pd
import numpy as np

import importlib
xgb_mod = importlib.import_module("src.models.01_xgboost_baseline")


# ---------------------------------------------------------------------------
# compute_extreme_threshold
# ---------------------------------------------------------------------------

class TestComputeExtremeThreshold:
    def test_threshold_isolates_outlier(self, dummy_weather_df):
        """The 99th percentile should sit well above the normal range."""
        threshold = xgb_mod.compute_extreme_threshold(dummy_weather_df)
        assert threshold > 5.0, "Threshold failed to isolate the rare extreme event."

    def test_custom_quantile(self, dummy_weather_df):
        """A lower quantile should produce a lower threshold."""
        t50 = xgb_mod.compute_extreme_threshold(dummy_weather_df, quantile=0.50)
        t99 = xgb_mod.compute_extreme_threshold(dummy_weather_df, quantile=0.99)
        assert t50 < t99

    def test_returns_float(self, dummy_weather_df):
        result = xgb_mod.compute_extreme_threshold(dummy_weather_df)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# create_target_column
# ---------------------------------------------------------------------------

class TestCreateTargetColumn:
    def test_adds_binary_column(self, dummy_weather_df):
        threshold = xgb_mod.compute_extreme_threshold(dummy_weather_df)
        result = xgb_mod.create_target_column(dummy_weather_df, threshold)
        assert 'is_extreme' in result.columns
        assert set(result['is_extreme'].unique()).issubset({0, 1})

    def test_flags_at_least_one_extreme(self, dummy_weather_df):
        threshold = xgb_mod.compute_extreme_threshold(dummy_weather_df)
        result = xgb_mod.create_target_column(dummy_weather_df, threshold)
        assert result['is_extreme'].sum() >= 1

    def test_no_mutation_of_input(self, dummy_weather_df):
        threshold = xgb_mod.compute_extreme_threshold(dummy_weather_df)
        xgb_mod.create_target_column(dummy_weather_df, threshold)
        assert 'is_extreme' not in dummy_weather_df.columns


# ---------------------------------------------------------------------------
# prepare_features
# ---------------------------------------------------------------------------

class TestPrepareFeatures:
    def test_no_leakage_precip(self, dummy_weather_df):
        """actual_precip_mm must NOT be in the feature set."""
        threshold = xgb_mod.compute_extreme_threshold(dummy_weather_df)
        df = xgb_mod.create_target_column(dummy_weather_df, threshold)
        X, y = xgb_mod.prepare_features(df)
        assert 'actual_precip_mm' not in X.columns

    def test_no_leakage_target(self, dummy_weather_df):
        """is_extreme must NOT be in the feature set."""
        threshold = xgb_mod.compute_extreme_threshold(dummy_weather_df)
        df = xgb_mod.create_target_column(dummy_weather_df, threshold)
        X, y = xgb_mod.prepare_features(df)
        assert 'is_extreme' not in X.columns

    def test_correct_feature_columns(self, dummy_weather_df):
        threshold = xgb_mod.compute_extreme_threshold(dummy_weather_df)
        df = xgb_mod.create_target_column(dummy_weather_df, threshold)
        X, y = xgb_mod.prepare_features(df)
        assert list(X.columns) == ['temp_celsius', 'pressure_hpa', 'wind_u_vector', 'wind_v_vector']

    def test_y_matches_is_extreme(self, dummy_weather_df):
        threshold = xgb_mod.compute_extreme_threshold(dummy_weather_df)
        df = xgb_mod.create_target_column(dummy_weather_df, threshold)
        X, y = xgb_mod.prepare_features(df)
        pd.testing.assert_series_equal(y, df['is_extreme'])


# ---------------------------------------------------------------------------
# compute_class_weight
# ---------------------------------------------------------------------------

class TestComputeClassWeight:
    def test_expected_ratio(self):
        y_train = pd.Series([0] * 99 + [1] * 1)
        ratio = xgb_mod.compute_class_weight(y_train)
        assert ratio == pytest.approx(99.0)

    def test_balanced_classes(self):
        y_train = pd.Series([0] * 50 + [1] * 50)
        ratio = xgb_mod.compute_class_weight(y_train)
        assert ratio == pytest.approx(1.0)

    def test_returns_float(self):
        y_train = pd.Series([0, 1])
        assert isinstance(xgb_mod.compute_class_weight(y_train), float)


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

class TestTrainModel:
    def test_model_trains_and_predicts(self, dummy_weather_df):
        """End-to-end: the model pipeline should produce valid predictions."""
        threshold = xgb_mod.compute_extreme_threshold(dummy_weather_df)
        df = xgb_mod.create_target_column(dummy_weather_df, threshold)
        X, y = xgb_mod.prepare_features(df)
        weight = xgb_mod.compute_class_weight(y)
        model = xgb_mod.train_model(X, y, weight)

        preds = model.predict(X)
        assert set(preds).issubset({0, 1})
        assert len(preds) == len(X)
