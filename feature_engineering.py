"""
Universal Feature Engineering for Cryptocurrency OHLCV Data
============================================================
Supports:
  - LSTM price prediction  (default, backward-compatible)
  - LSTM log-return prediction  (future)
  - XGBoost log-return prediction
  - Model 3  (placeholder — extend get_feature_sets() when ready)

Feature sets
------------
Universal (shared across models):
  full          – all generated features
  technical     – universal optimum (no model-specific features)
  technical_bp  – technical + buy_pressure (requires taker_buy column)
  minimal       – smallest universal subset

LSTM-dedicated:
  lstm_momentum
  lstm_volatility
  lstm_price     – price-specific (SMA ratios, BB — suited for price target)

XGBoost-dedicated:
  xgb_momentum
  xgb_volatility
  xgb_returns    – pure return-based (no price-level features)

Placeholders (add your own features):
  k_set1
  k_set2
  k_set3
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# ============================================================================
# CORE FEATURE GENERATION
# ============================================================================

def _rolling_percentile_rank(values, window: int, chunk: int = 50_000) -> np.ndarray:
    """
    Percentile rank (0-100) of the last value inside each trailing window.

    Equivalent to ``Series.rolling(window).apply(lambda x: (x <= x.iloc[-1]).mean() * 100)``
    (rows with fewer than `window` observations or any NaN inside the window are NaN),
    but evaluated on strided window views instead of one Python call per row.
    Chunking bounds the temporary (chunk x window) boolean matrix.
    """
    v = np.asarray(values, dtype=np.float64)
    out = np.full(len(v), np.nan, dtype=np.float64)
    if len(v) < window:
        return out
    windows = sliding_window_view(v, window)
    for start in range(0, len(windows), chunk):
        block = windows[start:start + chunk]
        pct = (block <= block[:, -1:]).mean(axis=1) * 100.0
        pct[np.isnan(block).any(axis=1)] = np.nan
        first = window - 1 + start
        out[first:first + len(block)] = pct
    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all technical features from raw OHLCV data.

    Args:
        df: DataFrame with columns: open_time, open, high, low, close, volume
            Optionally: taker_buy_base_asset_volume  (for buy_pressure)

    Returns:
        DataFrame with additional feature columns appended.
        Original columns are preserved unchanged.
    """
    df = df.copy()

    # -------------------------------------------------------------------------
    # LOG-RETURNS  (multiple timeframes)
    # -------------------------------------------------------------------------
    for window in [1, 5, 15, 60]:
        df[f'return_{window}'] = np.log(df['close'] / df['close'].shift(window))

    # -------------------------------------------------------------------------
    # SMA RATIOS  (close / SMA — normalised price position)
    # -------------------------------------------------------------------------
    for window in [7, 14, 30]:
        sma = df['close'].rolling(window).mean()
        df[f'close_sma_{window}'] = df['close'] / sma

    # -------------------------------------------------------------------------
    # MACD
    # -------------------------------------------------------------------------
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # -------------------------------------------------------------------------
    # RSI  (bugfix: clip loss to avoid Inf/NaN propagation when loss == 0)
    # -------------------------------------------------------------------------
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0).rolling(14).mean()).clip(lower=1e-10)
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # -------------------------------------------------------------------------
    # VOLATILITY  (rolling std of return_1)
    # -------------------------------------------------------------------------
    for window in [3, 7, 14, 30]:
        df[f'volatility_{window}'] = df['return_1'].rolling(window).std()

    # -------------------------------------------------------------------------
    # VOLATILITY REGIME
    # z-score of 30-bar realised vol over a 500-bar rolling window.
    # Tells the model where current volatility sits on its historical
    # distribution — value 0.0 = median regime, +2.0 = elevated, -1.0 = calm.
    # Avoids raw magnitude differences across market regimes.
    # Note: requires ~530 bars to be non-NaN; NaN rows are dropped by callers.
    # -------------------------------------------------------------------------
    realized_vol = df['return_1'].rolling(30).std()
    vol_roll_mean = realized_vol.rolling(500).mean()
    vol_roll_std = realized_vol.rolling(500).std().clip(lower=1e-10)
    df['vol_regime'] = (realized_vol - vol_roll_mean) / vol_roll_std

    # -------------------------------------------------------------------------
    # VOLUME
    # -------------------------------------------------------------------------
    volume_sma = df['volume'].rolling(20).mean()
    df['relative_volume'] = df['volume'] / volume_sma.clip(lower=1e-10)

    # -------------------------------------------------------------------------
    # BUY PRESSURE  (order flow imbalance)
    # taker_buy_base_asset_volume / total_volume  in [0, 1].
    # Order flow imbalance — potentially informative for short-term prediction.
    # Empirical FI ranking: rank 20/23 in LR (ΔMAE +0.00004),
    #                       rank 14/29 in Signal (ΔMAE +0.00058).
    # Low-to-moderate impact in both modules.
    # Gracefully skipped when the column is absent (non-Binance sources).
    # -------------------------------------------------------------------------
    if 'taker_buy_base_asset_volume' in df.columns:
        df['buy_pressure'] = (
            df['taker_buy_base_asset_volume'] / df['volume'].clip(lower=1e-10)
        )

    # -------------------------------------------------------------------------
    # BOLLINGER BANDS
    # bugfix: clip denominator to avoid 0/0 = NaN when all 20 candles identical
    # -------------------------------------------------------------------------
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    df['bb_position'] = (df['close'] - bb_lower) / (
        (bb_upper - bb_lower).clip(lower=1e-10)
    )

    # -------------------------------------------------------------------------
    # HIGH-LOW RANGE
    # -------------------------------------------------------------------------
    df['hl_range'] = (df['high'] - df['low']) / df['close'].clip(lower=1e-10)

    # -------------------------------------------------------------------------
    # TIME FEATURES  (cyclical encoding)
    # Normalise open_time before extracting hour/dow:
    #   - strip timezone info if present  (e.g. '2026-01-06 23:44:00+00:00')
    #   - handles both tz-aware and tz-naive strings uniformly
    # Without this, dt.dt.hour on a tz-aware Series returns NaN in some
    # pandas versions, causing hour_sin/cos and dow_sin/cos to be all-NaN
    # which then fails sklearn's feature-name check after dropna removes rows.
    # -------------------------------------------------------------------------
    dt = pd.to_datetime(df['open_time'], utc=True).dt.tz_localize(None)
    # dt = pd.to_datetime(df['open_time'])
    hour = dt.dt.hour
    dow = dt.dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # -------------------------------------------------------------------------
    # MULTI-TIMEFRAME / REGIME (Faza 4a — additive, does not alter above)
    # -------------------------------------------------------------------------
    df['return_15m_agg'] = df['close'].pct_change(15)
    df['return_1h_agg'] = df['close'].pct_change(60)
    df['volatility_1h'] = df['return_1'].rolling(60).std()
    gain15 = delta.where(delta > 0, 0).rolling(15).mean()
    loss15 = (-delta.where(delta < 0, 0).rolling(15).mean()).clip(lower=1e-10)
    rs15 = gain15 / loss15
    df['rsi_15m'] = 100 - (100 / (1 + rs15))
    # ADX-14 simplified
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().clip(lower=1e-10)
    plus_dm = (df['high'].diff()).clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    plus_di = 100 * plus_dm.rolling(14).mean() / atr14
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100
    df['adx_14'] = dx.rolling(14).mean()
    # Efficiency ratio (30 bars)
    net_change = (df['close'] - df['close'].shift(30)).abs()
    path_sum = df['close'].diff().abs().rolling(30).sum().clip(lower=1e-10)
    df['efficiency_ratio_30'] = net_change / path_sum
    if 'volatility_14' in df.columns:
        df['vol_percentile_240'] = _rolling_percentile_rank(
            df['volatility_14'].to_numpy(), 240)

    # -------------------------------------------------------------------------
    # NEW FEATURES (Analysis #3 — Feature Engineering Audit 2026-08)
    # -------------------------------------------------------------------------

    # ATR-14 normalised by close price (atr14 already computed above in MTF)
    df['atr_14_norm'] = atr14 / df['close'].clip(lower=1e-10)

    # Candle pattern features (normalised OHLCV relationships)
    hl_diff = (df['high'] - df['low']).clip(lower=1e-10)
    df['candle_body_ratio'] = (df['close'] - df['open']).abs() / hl_diff
    df['close_position'] = (df['close'] - df['low']) / hl_diff
    df['upper_shadow'] = (
        df['high'] - df[['close', 'open']].max(axis=1)
    ) / hl_diff

    # Momentum acceleration (5-bar change of return_1)
    df['momentum_accel'] = df['return_1'].diff(5)

    # Volume trend (short-term vs medium-term volume ratio)
    vol_sma_14 = df['volume'].rolling(14).mean()
    vol_sma_60 = df['volume'].rolling(60).mean().clip(lower=1e-10)
    df['volume_trend'] = vol_sma_14 / vol_sma_60

    # VWAP position proxy (close relative to rolling VWAP)
    vwap_proxy = (
        (df['close'] * df['volume']).rolling(20).sum()
        / df['volume'].rolling(20).sum().clip(lower=1e-10)
    )
    df['vwap_position'] = df['close'] / vwap_proxy.clip(lower=1e-10)

    # Distance to threshold in sigma units (from audit §6.2).
    # Return normalised by backward-looking rolling sigma — same scale
    # as Signal oracle threshold T.  Uses 20-bar window (default WINDOW_SIGMA).
    # Only backward-looking data → no leakage.
    rolling_sigma = df['return_1'].rolling(20).std().clip(lower=1e-10)
    df['dist_to_threshold_sigma'] = df['return_1'] / rolling_sigma

    return df


# ============================================================================
# FEATURE COLUMN REGISTRY
# ============================================================================

def get_feature_columns(include_buy_pressure: bool = False,
                        include_mtf: bool = False) -> list:
    """
    Canonical ordered list of all generated feature names.

    Args:
        include_buy_pressure: Include 'buy_pressure' only when the source data
            contained taker_buy_base_asset_volume.  Default False keeps the
            list deterministic for callers that do not have that column.
        include_mtf: Include multi-timeframe, regime and Analysis #3 features.
            Default False for backward compatibility with existing callers.
            Set True to get the full superset of all features generated by
            prepare_features().

    Returns:
        list of feature names (raw OHLCV / Binance passthrough columns excluded).
    """
    features = []

    # Returns
    features.extend([f'return_{w}' for w in [1, 5, 15, 60]])

    # Trend / momentum
    features.extend([f'close_sma_{w}' for w in [7, 14, 30]])
    features.extend(['macd', 'macd_signal', 'macd_diff', 'rsi'])

    # Volatility (legacy [7, 14, 30] preserved for backward-compatibility)
    features.extend([f'volatility_{w}' for w in [7, 14, 30]])
    features.append('vol_regime')

    # Volume / microstructure
    features.append('relative_volume')
    if include_buy_pressure:
        features.append('buy_pressure')

    # Price pattern
    features.extend(['bb_position', 'hl_range'])

    # Time
    features.extend(['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'])

    if include_mtf:
        # Volatility 3m
        features.append('volatility_3')
        # Multi-timeframe / regime (Faza 4a)
        features.extend([
            'return_15m_agg', 'return_1h_agg', 'volatility_1h', 'rsi_15m',
            'adx_14', 'efficiency_ratio_30', 'vol_percentile_240',
        ])
        # Analysis #3 new features
        features.extend([
            'atr_14_norm',
            'candle_body_ratio', 'close_position', 'upper_shadow',
            'momentum_accel', 'volume_trend', 'vwap_position',
            'dist_to_threshold_sigma',
        ])

    return features



# ============================================================================
# FEATURE SETS
# ============================================================================

def _merge_sets(*lists) -> list:
    """Merge feature lists, preserving order, removing duplicates."""
    seen = set()
    result = []
    for lst in lists:
        for feat in lst:
            if feat not in seen:
                seen.add(feat)
                result.append(feat)
    return result


def get_feature_sets(include_buy_pressure: bool = False) -> dict:
    """
    All named feature sets.

    Structure (Analysis #3, 2026-08):
      - Legacy sets: full, technical, technical_bp, technical_mtf, minimal,
                     lstm_*, xgb_*, k_set* — UNCHANGED for backward compat.
      - New superset: max — true superset of all features (incl. MTF + new).
      - FI-based: lr_core, lr_optimal, signal_core, signal_optimal —
                  built from permutation importance rankings.
      - Aggregator: aggregator_ext, aggregator_overlap, aggregator_disjoint —
                    three approaches for Signal model with LR submodels.
      - Probe: lr_neutral_probe, signal_neutral_probe — diagnostic sets
               for testing the "masking" hypothesis.
      - Composable: trend_momentum, volatility_regime, core_combined —
                    unions of atomic feature blocks.

    Args:
        include_buy_pressure: When True, 'buy_pressure' is included in sets
            that benefit from microstructure signals (full, technical_bp,
            xgb_momentum, lstm_momentum, and derived composites).
            Pass True only when raw data contains taker_buy_base_asset_volume.

    Returns:
        dict  set_name -> list[str]
    """
    all_features = get_feature_columns(include_buy_pressure=include_buy_pressure)

    # ==================================================================
    # ATOMIC FEATURE BLOCKS
    # (internal building blocks — not exposed as sets directly)
    # ==================================================================

    _returns_short = ['return_1', 'return_5', 'return_15']
    _returns_ext = ['return_1', 'return_5', 'return_15', 'return_60']
    _trend_sma = ['close_sma_7', 'close_sma_14', 'close_sma_30']
    _macd_triplet = ['macd', 'macd_signal', 'macd_diff']
    _oscillators = ['rsi', 'bb_position']
    _volatility_short = ['volatility_7', 'volatility_14']
    _volatility_ext = ['volatility_3', 'volatility_7', 'volatility_14',
                        'volatility_30', 'vol_regime']
    _volume_micro = ['relative_volume']
    if include_buy_pressure:
        _volume_micro = ['relative_volume', 'buy_pressure']
    _price_patterns = ['bb_position', 'hl_range']
    _seasonality = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    _mtf_features = [
        'return_15m_agg', 'return_1h_agg', 'volatility_1h', 'rsi_15m',
        'adx_14', 'efficiency_ratio_30', 'vol_percentile_240',
    ]
    _new_candle = ['candle_body_ratio', 'close_position', 'upper_shadow']
    _new_derived = ['atr_14_norm', 'momentum_accel', 'volume_trend',
                    'vwap_position']
    _new_signal = ['dist_to_threshold_sigma']
    _new_volatility = ['volatility_3']

    # ==================================================================
    # LEGACY SETS (UNCHANGED — backward compatibility)
    # ==================================================================

    # full — every generated feature (canonical reference set)
    # NOTE: does NOT include MTF features — this is intentional for
    # backward compatibility.  Use 'max' for the true superset.
    full = all_features

    # technical — balanced universal set
    #   Excludes:
    #     • time features (hour/dow) — not always useful; add via lstm_* sets
    #     • vol_regime — 500-bar warm-up window; excluded to keep 'technical'
    #       safe for shorter prepared-data windows
    #   Includes: returns, SMA ratios, MACD, RSI, volatility, volume, BB, HL
    technical = [
        'return_1', 'return_5', 'return_15',
        'close_sma_7', 'close_sma_14', 'close_sma_30',
        'macd', 'macd_signal', 'macd_diff',
        'rsi',
        'volatility_7', 'volatility_14',
        'relative_volume',
        'bb_position',
        'hl_range',
    ]

    # technical_bp — technical + buy_pressure (order-flow imbalance)
    # Requires taker_buy_base_asset_volume in raw data (Binance full export).
    technical_bp = technical.copy()
    if include_buy_pressure:
        technical_bp.append('buy_pressure')

    mtf_features = _mtf_features
    technical_mtf = technical_bp + mtf_features

    # minimal — smallest viable universal subset
    minimal = [
        'return_1', 'return_5',
        'close_sma_14',
        'rsi',
        'volatility_14',
        'relative_volume',
    ]

    # ------------------------------------------------------------------
    # LSTM-DEDICATED SETS (UNCHANGED)
    # ------------------------------------------------------------------

    lstm_momentum = [
        'return_1', 'return_5', 'return_15', 'return_60',
        'macd', 'macd_signal', 'macd_diff',
        'rsi',
        'close_sma_7', 'close_sma_14', 'close_sma_30',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    ]
    if include_buy_pressure:
        lstm_momentum.append('buy_pressure')

    lstm_volatility = [
        'return_1', 'return_5',
        'volatility_7', 'volatility_14', 'volatility_30',
        'vol_regime',
        'bb_position',
        'hl_range',
        'relative_volume',
        'hour_sin', 'hour_cos',
    ]

    lstm_price = [
        'return_1', 'return_5', 'return_15',
        'close_sma_7', 'close_sma_14', 'close_sma_30',
        'bb_position',
        'rsi',
        'macd', 'macd_diff',
        'volatility_14',
        'relative_volume',
        'hl_range',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    ]

    # ------------------------------------------------------------------
    # XGBOOST-DEDICATED SETS (UNCHANGED)
    # ------------------------------------------------------------------

    xgb_momentum = [
        'return_1', 'return_5', 'return_15', 'return_60',
        'macd', 'macd_signal', 'macd_diff',
        'rsi',
        'close_sma_7', 'close_sma_14', 'close_sma_30',
        'vol_regime',
    ]
    if include_buy_pressure:
        xgb_momentum.append('buy_pressure')

    xgb_volatility = [
        'return_1', 'return_5',
        'volatility_7', 'volatility_14', 'volatility_30',
        'vol_regime',
        'bb_position',
        'hl_range',
        'relative_volume',
    ]

    xgb_returns = [f'return_{w}' for w in [1, 5, 15, 60]]

    # ------------------------------------------------------------------
    # PLACEHOLDERS (UNCHANGED)
    # ------------------------------------------------------------------
    k_set1: list = []
    k_set2: list = []
    k_set3: list = []

    # ==================================================================
    # NEW SETS (Analysis #3)
    # ==================================================================

    # --- SUPERSET ---
    # max — true superset of ALL features generated by prepare_features().
    # Use this for new dataset preparation (--feature_set max).
    max_set = get_feature_columns(
        include_buy_pressure=include_buy_pressure, include_mtf=True)

    # --- FI-BASED SETS (from permutation importance analysis) ---
    # lr_core: Top 7 features by mean ΔMAE in 714 LR models
    # (macd_signal, macd, bb_position, macd_diff, volatility_7/14, return_5)
    lr_core = _merge_sets(
        _macd_triplet,
        _volatility_short,
        ['return_5', 'bb_position'],
    )

    # lr_optimal: lr_core + strong secondary features for LR
    lr_optimal = _merge_sets(
        lr_core,
        _trend_sma,
        ['return_15', 'hl_range', 'rsi', 'volatility_30'],
    )

    # signal_core: Top 6 features by mean ΔMAE in 171 Signal models
    # (volatility_14, rsi, close_sma_14, hl_range, macd, macd_signal)
    signal_core = [
        'volatility_14', 'hl_range', 'rsi', 'close_sma_14',
        'macd', 'macd_signal',
    ]

    # signal_optimal: signal_core + strong secondary features for Signal
    signal_optimal = _merge_sets(
        signal_core,
        ['close_sma_7', 'close_sma_30', 'bb_position', 'return_1'],
    )

    # --- AGGREGATOR SETS (for Signal model with LR submodels) ---
    # Per-symbol submodel feature sets matching what LR submodels were trained on:
    # ETH submodels (e.g. LR1107, LR1041) were trained on E104 with 'full' (23 features with BP)
    # BTC submodels (e.g. LR1174, LR1173) were trained on B104 with 'technical_bp' (16 features)
    submodel_compat_eth = full.copy()
    submodel_compat_btc = technical_bp.copy()
    submodel_compat = submodel_compat_btc.copy()

    # aggregator_ext: EXTENDING approach (recommended)
    # submodel_compat + MTF + new derived features → no zeroing + new info
    aggregator_ext = _merge_sets(
        submodel_compat_btc,
        _mtf_features,
        _new_derived,
        _new_signal,
    )

    # aggregator_overlap: SHARED approach (control baseline)
    # Same features as submodel — purely tests fusion value
    aggregator_overlap = submodel_compat.copy()

    # aggregator_disjoint: DISJOINT approach (experimental)
    # Only complementary features, NO submodel features → requires raising
    # max_missing_ratio in SubmodelManager (--submodel_max_missing_ratio 0.9)
    aggregator_disjoint = _merge_sets(
        _mtf_features,
        ['volatility_14', 'hl_range', 'rsi', 'close_sma_14'],
        _new_derived,
        _new_signal,
    )

    # --- PROBE SETS (diagnostic — neutral feature hypothesis testing) ---
    # lr_neutral_probe: features with near-zero or negative ΔMAE in LR
    # Tests masking hypothesis: are these truly useless, or masked by MACD?
    lr_neutral_probe = [
        'relative_volume', 'return_60', 'rsi', 'return_1',
    ]
    if include_buy_pressure:
        lr_neutral_probe.append('buy_pressure')

    # signal_neutral_probe: features with near-zero or negative ΔMAE in Signal
    # Tests masking hypothesis: are these masked by volatility_14/hl_range?
    signal_neutral_probe = [
        'return_5', 'volatility_7', 'return_60',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'relative_volume',
    ]

    # --- COMPOSABLE BLOCK UNIONS ---
    # trend_momentum: trend + momentum family
    trend_momentum = _merge_sets(_trend_sma, _macd_triplet)

    # volatility_regime: all volatility + price pattern features
    volatility_regime = _merge_sets(_volatility_ext, _price_patterns)

    # core_combined: union of lr_core + signal_core (deduplicated)
    core_combined = _merge_sets(lr_core, signal_core)

    # full_extended: full + all new Analysis #3 features (no MTF)
    full_extended = _merge_sets(
        full, _new_candle, _new_derived, _new_signal, _new_volatility)

    # candle_plus_bp: technical_bp + candle pattern features
    candle_plus_bp = _merge_sets(technical_bp, _new_candle)

    # submodel_compat FE 3.0 aliases (CP-146)
    # ETH FE 3.0 models (e.g. LR1205) match full_extended 1:1 (32 features)
    submodel_compat_eth_fe3 = full_extended.copy()
    # Provisional BTC FE 3.0 draft by architectural analogy (25 features)
    submodel_compat_btc_fe3 = _merge_sets(
        technical_bp, _new_candle, _new_derived, _new_signal, _new_volatility
    )

    # --- FUTURE / DOCUMENTATION ---
    # raw_ohlcv: surowe kolumny OHLCV jako cechy (eksperymentalne).
    # OHLCV jest dostępne w CSV (_save_csv_split() dodaje base_ohlcv).
    # Odkomentuj i dodaj do return dict gdy gotowe do testowania.
    # Wymaga filtracji duplikacji w prepare_data.py (linia 107).
    # _raw_ohlcv = ['open', 'high', 'low', 'close', 'volume']

    return {
        # Universal (LEGACY — unchanged)
        'full':             full,
        'technical':        technical,
        'technical_bp':     technical_bp,
        'technical_mtf':    technical_mtf,
        'minimal':          minimal,

        # LSTM-dedicated (LEGACY — unchanged)
        'lstm_momentum':    lstm_momentum,
        'lstm_volatility':  lstm_volatility,
        'lstm_price':       lstm_price,

        # XGBoost-dedicated (LEGACY — unchanged)
        'xgb_momentum':     xgb_momentum,
        'xgb_volatility':   xgb_volatility,
        'xgb_returns':      xgb_returns,

        # Placeholders (LEGACY — unchanged)
        'k_set1':           k_set1,
        'k_set2':           k_set2,
        'k_set3':           k_set3,

        # NEW — Superset
        'max':              max_set,

        # NEW — FI-based
        'lr_core':          lr_core,
        'lr_optimal':       lr_optimal,
        'signal_core':      signal_core,
        'signal_optimal':   signal_optimal,

        # NEW — Aggregator variants (for submodel experiments)
        'submodel_compat':      submodel_compat,
        'submodel_compat_eth':  submodel_compat_eth,
        'submodel_compat_btc':  submodel_compat_btc,
        'submodel_compat_eth_fe3': submodel_compat_eth_fe3,
        'submodel_compat_btc_fe3': submodel_compat_btc_fe3,
        'aggregator_ext':       aggregator_ext,
        'aggregator_overlap':   aggregator_overlap,
        'aggregator_disjoint':  aggregator_disjoint,

        # NEW — Probe / diagnostic
        'lr_neutral_probe':     lr_neutral_probe,
        'signal_neutral_probe': signal_neutral_probe,

        # NEW — Composable block unions
        'trend_momentum':   trend_momentum,
        'volatility_regime': volatility_regime,
        'core_combined':    core_combined,
        'full_extended':    full_extended,
        'candle_plus_bp':   candle_plus_bp,
    }


def get_submodel_compat_set(symbol: str = 'ETH',
                            include_buy_pressure: bool = True,
                            fe_version: str | int = 'legacy') -> list:
    """
    Get the appropriate submodel feature list for a given asset symbol.

    Args:
        symbol: Asset symbol ('ETH', 'BTC', etc.)
        include_buy_pressure: Forwarded to get_feature_sets()
        fe_version: 'legacy' (default, 23 feats for ETH, 16 for BTC) or
            'fe3' / 3 (32 feats for ETH FE3.0)

    Returns:
        list of feature names expected by submodels of that asset
    """
    sets = get_feature_sets(include_buy_pressure=include_buy_pressure)
    sym = (symbol or 'ETH').upper()
    is_fe3 = str(fe_version).lower() in ('3', 'fe3', 'v3')
    if 'BTC' in sym:
        if is_fe3:
            return sets.get('submodel_compat_btc_fe3', sets['technical_bp'])
        return sets.get('submodel_compat_btc', sets['technical_bp'])
    if is_fe3:
        return sets.get('submodel_compat_eth_fe3', sets['full_extended'])
    return sets.get('submodel_compat_eth', sets['full'])


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_features(
    df: pd.DataFrame,
    feature_set: str = None,
    custom_features: list = None,
    validate: bool = True,
    include_buy_pressure: bool = False,
) -> list:
    """
    Select features from a DataFrame.

    Priority order: custom_features > feature_set > default (full).

    Args:
        df:                  DataFrame after prepare_features() has been called.
        feature_set:         Name of a predefined set.  See get_feature_sets().
        custom_features:     Explicit list — overrides feature_set entirely.
        validate:            True  → raise ValueError for any missing columns.
                             False → silently drop missing columns.
        include_buy_pressure: Forwarded to get_feature_sets() /
                              get_feature_columns() when building the candidate
                              list.  Ignored when custom_features is provided.

    Returns:
        list of feature column names.
    """
    if custom_features is not None:
        if validate:
            missing = [f for f in custom_features if f not in df.columns]
            if missing:
                raise ValueError(
                    f"Missing custom features in DataFrame: {missing}"
                )
            return list(custom_features)
        return [f for f in custom_features if f in df.columns]

    if feature_set is not None:
        sets = get_feature_sets(include_buy_pressure=include_buy_pressure)
        if feature_set not in sets:
            raise ValueError(
                f"Unknown feature set: '{feature_set}'. "
                f"Available: {sorted(sets.keys())}"
            )
        features = sets[feature_set]
        if not features:
            # Placeholder set (k_set*) — return empty list without raising
            return []
        if validate:
            missing = [f for f in features if f not in df.columns]
            if missing:
                raise ValueError(
                    f"Features missing from DataFrame for set "
                    f"'{feature_set}': {missing}"
                )
            return list(features)
        return [f for f in features if f in df.columns]

    # Default: full feature list
    features = get_feature_columns(include_buy_pressure=include_buy_pressure)
    if validate:
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing standard features: {missing}")
        return list(features)
    return [f for f in features if f in df.columns]


# ============================================================================
# BINANCE COLUMN REGISTRY
# ============================================================================

def get_binance_columns() -> list:
    """
    Binance API passthrough columns that are NOT features and should be
    excluded from feature DataFrames.

    Design note: taker_buy_base_asset_volume is intentionally kept in this
    list so it is never accidentally used as a raw feature.  prepare_features()
    consumes it to produce the derived 'buy_pressure' feature first; callers
    that strip Binance columns afterwards will then drop the raw source column
    while keeping the engineered one.
    """
    return [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
        'ignore',
        # Alternative capitalisation variants (some Binance export formats)
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume',
        'Ignore',
    ]
