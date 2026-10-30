"""
Central configuration for the LSTM price prediction pipeline.

Defines simulation parameters, fee structures, feature sets, ranking constants,
and storage path conventions.
"""
from __future__ import annotations

import os
from pathlib import Path

# Base Paths (relative to repository root)
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

MODELS_DIR = BASE_DIR / 'lstm_models'
DATA_DIR = BASE_DIR / 'lstm_data'
RESULTS_DIR = DATA_DIR
SCRIPTS_DIR = BASE_DIR / 'scripts'
PREPARED_DATA_DIR = REPO_ROOT / 'data_preparation' / 'prepared_data'

# Simulation Profile v2 Constants
SIM_START_CASH = 100.0
SIM_PRIMARY_FEE_BPS = 2.0
SIM_AUXILIARY_FEE_BPS = 2.0
SIM_GROSS_FEE_BPS = 0.0

SIGNAL_WINDOW_SIGMA = 100
SIGNAL_THRESHOLD_ECONOMIC = 0.001
SIGNAL_MIN_STRONG_DEFAULT = 60.0
WARMUP_DEFAULT = 2000
POSITION_MODE_DEFAULT = 'sticky'

# Ranking Standards
RANKING_METHOD_DEFAULT = 'absolute_60_gross'
SIMULATION_PROFILE_DEFAULT = 'lstm_simulation_v2'
RANK_METRIC_VERSION = 2

# Supported Feature Sets
# 'max' represents the full universal 39-feature set in E109; 'lr_optimal' and
# 'signal_optimal' match the dedicated E114 / E115 datasets; 'custom' takes an explicit
# column list from --custom_features.
ALL_FEATURE_SETS = [
    'max',
    'lr_optimal',
    'signal_optimal',
    'custom',
    'full',
    'technical',
    'technical_bp',
    'minimal',
    'lstm_momentum',
    'lstm_volatility',
    'lstm_price',
    'xgb_momentum',
    'xgb_volatility',
    'xgb_returns',
    'k_set1',
    'k_set2',
    'k_set3',
]

# Universal 39 Features from E109 Metadata
UNIVERSAL_MAX_39_FEATURES = [
    "return_1",
    "return_5",
    "return_15",
    "return_60",
    "close_sma_7",
    "close_sma_14",
    "close_sma_30",
    "macd",
    "macd_signal",
    "macd_diff",
    "rsi",
    "volatility_3",
    "volatility_7",
    "volatility_14",
    "volatility_30",
    "vol_regime",
    "relative_volume",
    "buy_pressure",
    "bb_position",
    "hl_range",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "return_15m_agg",
    "return_1h_agg",
    "volatility_1h",
    "rsi_15m",
    "adx_14",
    "efficiency_ratio_30",
    "vol_percentile_240",
    "atr_14_norm",
    "candle_body_ratio",
    "close_position",
    "upper_shadow",
    "momentum_accel",
    "volume_trend",
    "vwap_position",
    "dist_to_threshold_sigma",
]

# Training Defaults
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 256
DEFAULT_LOOKBACK = 48
DEFAULT_HORIZON = 12
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 1
DEFAULT_DROPOUT = 0.1
DEFAULT_LR = 0.001
DEFAULT_SEED = 42
