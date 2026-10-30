"""
XGBoost Model Adapter — CSV-based (no NPY)
"""
import os
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    FunctionTransformer,
)

from model_adapters.feature_engineering import prepare_features
from model_adapters.feature_engineering import select_features as _select_features


def get_scaler(scaler_type):
    """Get scaler instance"""
    if scaler_type is None or str(scaler_type).lower() in ('none', 'raw', 'noop'):
        return FunctionTransformer(func=None, inverse_func=None, validate=False)
    st = str(scaler_type).lower()
    scalers = {
        'minmax': MinMaxScaler(),
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'quantile_normal': QuantileTransformer(output_distribution='normal', random_state=42),
        'quantile_uniform': QuantileTransformer(output_distribution='uniform', random_state=42),
        'none': FunctionTransformer(func=None, inverse_func=None, validate=False),
        'raw': FunctionTransformer(func=None, inverse_func=None, validate=False),
    }
    return scalers.get(st, StandardScaler())


def select_features(df, feature_set='full', custom_features=None,
                    include_buy_pressure=False):
    """
    Select features using the universal feature engineering module.

    Args:
        df:                  DataFrame after prepare_features().
        feature_set:         Name of a predefined set (see feature_engineering.py).
        custom_features:     Explicit list — overrides feature_set.
        include_buy_pressure: Set True when raw data contains
                              taker_buy_base_asset_volume so that
                              'buy_pressure' is included in full / technical_bp
                              and *_momentum sets.
    """
    if custom_features:
        return custom_features
    return _select_features(
        df,
        feature_set=feature_set,
        validate=False,
        include_buy_pressure=include_buy_pressure,
    )


def prepare_for_xgb(train_df, agent_df, test_df, sym_df, features, lookback, horizon,
                    scaler_features='standard'):
    """
    Prepare data for XGBoost (fit feature scaler only — no sequences, no targets).

    XGBoost loads CSV files and performs flattening on-the-fly in xgb_model.
    This function only fits the scaler on train data to prevent leakage.
    """
    feature_scaler = get_scaler(scaler_features)
    feature_scaler.fit(train_df[features])

    return {
        'feature_scaler': feature_scaler,
        'base_features': features,
    }


def save_xgb_data(output_dir, data):
    """XGBoost uses CSV files — no NPY saving needed here."""
    pass
