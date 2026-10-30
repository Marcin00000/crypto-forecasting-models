"""
LSTM Model Adapter
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


def create_sequences(data, features, lookback, horizon):
    """Create 3D sequences for LSTM"""
    data_array = data[features].values.astype(np.float32)
    prices = data['close'].values.astype(np.float32)

    n_samples = len(data) - lookback - horizon + 1
    if n_samples <= 0:
        return None, None

    X = np.zeros((n_samples, lookback, len(features)), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        X[i] = data_array[i:i + lookback]
        y[i] = prices[i + lookback + horizon - 1]

    return X, y


def prepare_for_lstm(train_df, agent_df, test_df, sym_df, features, lookback, horizon,
                     scaler_features='standard', scaler_price='standard', use_log_prices=True):
    """Prepare data in LSTM format"""
    feature_scaler = get_scaler(scaler_features)
    price_scaler = get_scaler(scaler_price)

    # Fit on train only
    feature_scaler.fit(train_df[features])

    if use_log_prices:
        train_df['close'] = np.log(train_df['close'])
    price_scaler.fit(train_df['close'].values.reshape(-1, 1))

    # Transform train
    train_df[features] = feature_scaler.transform(train_df[features])
    train_df['close'] = price_scaler.transform(train_df['close'].values.reshape(-1, 1)).flatten()

    X_train, y_train = create_sequences(train_df, features, lookback, horizon)

    # Transform and create sequences for test
    if use_log_prices:
        test_df['close'] = np.log(test_df['close'])
    test_df[features] = feature_scaler.transform(test_df[features])
    test_df['close'] = price_scaler.transform(test_df['close'].values.reshape(-1, 1)).flatten()

    X_test, y_test = create_sequences(test_df, features, lookback, horizon)

    # Transform agent split (agent.csv) and sym (2D array format, no sequences)
    agent_data = None
    if agent_df is not None and len(agent_df) > 0:
        agent_df[features] = feature_scaler.transform(agent_df[features])
        if use_log_prices:
            agent_df['close'] = np.log(agent_df['close'])
        agent_df['close'] = price_scaler.transform(agent_df['close'].values.reshape(-1, 1)).flatten()
        agent_data = agent_df[features + ['close']].values

    sym_data = None
    if sym_df is not None and len(sym_df) > 0:
        sym_df[features] = feature_scaler.transform(sym_df[features])
        if use_log_prices:
            sym_df['close'] = np.log(sym_df['close'])
        sym_df['close'] = price_scaler.transform(sym_df['close'].values.reshape(-1, 1)).flatten()
        sym_data = sym_df[features + ['close']].values

    # Create 2D test data
    test_2d = test_df[features + ['close']].values.astype(np.float32)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'test_2d': test_2d,
        'agent_data': agent_data,
        'sym_data': sym_data,
        'feature_scaler': feature_scaler,
        'price_scaler': price_scaler,
        'use_log_prices': use_log_prices,
    }


def save_lstm_data(output_dir, data, prefix=''):
    """Save LSTM data in native format"""
    if prefix:
        prefix = prefix + '_'

    np.save(os.path.join(output_dir, f'{prefix}X_train.npy'), data['X_train'])
    np.save(os.path.join(output_dir, f'{prefix}y_train.npy'), data['y_train'])
    np.save(os.path.join(output_dir, f'{prefix}X_test_3d.npy'), data['X_test'])
    np.save(os.path.join(output_dir, f'{prefix}y_test_3d.npy'), data['y_test'])
    np.save(os.path.join(output_dir, f'{prefix}test_2d.npy'), data['test_2d'])

    if data.get('agent_data') is not None:
        np.save(os.path.join(output_dir, f'{prefix}agent_data.npy'), data['agent_data'])

    if data.get('sym_data') is not None:
        np.save(os.path.join(output_dir, f'{prefix}sym_data.npy'), data['sym_data'])

    import joblib
    joblib.dump(data['feature_scaler'], os.path.join(output_dir, 'scaler_features.pkl'))
    joblib.dump(data['price_scaler'], os.path.join(output_dir, 'scaler_price.pkl'))
