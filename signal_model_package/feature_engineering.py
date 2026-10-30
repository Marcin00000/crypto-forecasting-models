"""
Universal Feature Engineering for Cryptocurrency OHLCV Data.
Standalone implementation for signal_model_package supporting all 39 features from FE 3.0.
Optimised with vectorized rolling percentile (NumPy sliding_window_view).
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


# Prepared datasets built before vol_percentile_240 was corrected store the share of the
# window that is >= the last value (inverse percentile). Models trained on them were fitted
# to that definition, so inference must reproduce it; newer datasets use <= (true percentile).
LEGACY_VOL_PERCENTILE_DATA_IDS = frozenset({'E107', 'E108', 'E109'})


def uses_legacy_vol_percentile(data_id) -> bool:
    """True when a model trained on dataset `data_id` expects the legacy vol_percentile_240."""
    return str(data_id or '').strip().upper() in LEGACY_VOL_PERCENTILE_DATA_IDS


def get_required_warmup(features: list[str]) -> int:
    """Determine minimum candle warmup required by the feature set."""
    if 'vol_regime' in features:
        return 530
    if 'vol_percentile_240' in features:
        return 254
    return 100


def prepare_features(df: pd.DataFrame, features: Optional[list[str]] = None,
                     legacy_vol_percentile: bool = False) -> pd.DataFrame:
    """
    Generate technical features from raw OHLCV data.
    If features is provided, skips expensive calculations for unneeded features.

    Args:
        df: DataFrame with columns: open_time, open, high, low, close, volume.
            Optionally: taker_buy_base_asset_volume (for buy_pressure).
        features: Optional list of required feature names to compute selectively.
        legacy_vol_percentile: compute vol_percentile_240 with the legacy (>=) definition
            used by models trained on datasets E107-E109 (see uses_legacy_vol_percentile).

    Returns:
        DataFrame with feature columns appended.
    """
    df = df.copy()
    req = set(features) if features is not None else None

    # -------------------------------------------------------------------------
    # LOG-RETURNS (multiple timeframes)
    # -------------------------------------------------------------------------
    # return_1 is always needed as building block for volatility
    df['return_1'] = np.log(df['close'] / df['close'].shift(1))
    for window in [5, 15, 60]:
        if req is None or f'return_{window}' in req:
            df[f'return_{window}'] = np.log(df['close'] / df['close'].shift(window))

    # -------------------------------------------------------------------------
    # SMA RATIOS (close / SMA — normalised price position)
    # -------------------------------------------------------------------------
    for window in [7, 14, 30]:
        if req is None or f'close_sma_{window}' in req:
            sma = df['close'].rolling(window).mean()
            df[f'close_sma_{window}'] = df['close'] / sma

    # -------------------------------------------------------------------------
    # MACD
    # -------------------------------------------------------------------------
    if req is None or any(m in req for m in ('macd', 'macd_signal', 'macd_diff')):
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

    # -------------------------------------------------------------------------
    # RSI (clipped denominator to avoid Inf/NaN propagation)
    # -------------------------------------------------------------------------
    delta = None
    if req is None or any(r in req for r in ('rsi', 'rsi_15m')):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0).rolling(14).mean()).clip(lower=1e-10)
        rs = gain / loss
        df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

    # -------------------------------------------------------------------------
    # VOLATILITY (rolling std of return_1 for 3, 7, 14, 30)
    # -------------------------------------------------------------------------
    for window in [3, 7, 14, 30]:
        if req is None or f'volatility_{window}' in req or (window == 14 and ('vol_percentile_240' in req or 'vol_regime' in req)):
            df[f'volatility_{window}'] = df['return_1'].rolling(window).std()

    # -------------------------------------------------------------------------
    # VOLATILITY REGIME (z-score of 30-bar realised vol over 500-bar window)
    # -------------------------------------------------------------------------
    if req is None or 'vol_regime' in req:
        realized_vol = df['return_1'].rolling(30).std()
        vol_roll_mean = realized_vol.rolling(500).mean()
        vol_roll_std = realized_vol.rolling(500).std().clip(lower=1e-10)
        df['vol_regime'] = (realized_vol - vol_roll_mean) / vol_roll_std

    # -------------------------------------------------------------------------
    # VOLUME & BUY PRESSURE
    # -------------------------------------------------------------------------
    volume_sma = df['volume'].rolling(20).mean()
    df['relative_volume'] = df['volume'] / volume_sma.clip(lower=1e-10)

    if 'taker_buy_base_asset_volume' in df.columns:
        df['buy_pressure'] = (
            df['taker_buy_base_asset_volume'] / df['volume'].clip(lower=1e-10)
        )

    # -------------------------------------------------------------------------
    # BOLLINGER BANDS & HIGH-LOW RANGE
    # -------------------------------------------------------------------------
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    df['bb_position'] = (df['close'] - bb_lower) / (
        (bb_upper - bb_lower).clip(lower=1e-10)
    )
    df['hl_range'] = (df['high'] - df['low']) / df['close'].clip(lower=1e-10)

    # -------------------------------------------------------------------------
    # TIME FEATURES (cyclical encoding, timezone-normalized, auto timestamp detection)
    # -------------------------------------------------------------------------
    if 'open_time' in df.columns:
        col = df['open_time']
        num_col = pd.to_numeric(col, errors='coerce')
        if not num_col.isna().all():
            max_val = float(num_col.max()) if len(num_col) > 0 else 0.0
            if max_val > 1e16:
                unit = 'ns'
            elif max_val > 1e13:
                unit = 'us'
            elif max_val > 1e10:
                unit = 'ms'
            else:
                unit = 's'
            dt = pd.to_datetime(num_col, unit=unit, utc=True).dt.tz_localize(None)
        else:
            dt = pd.to_datetime(col, utc=True).dt.tz_localize(None)
        hour = dt.dt.hour
        dow = dt.dt.dayofweek
        df['hour_sin'] = np.sin(2.0 * np.pi * hour / 24.0)
        df['hour_cos'] = np.cos(2.0 * np.pi * hour / 24.0)
        df['dow_sin'] = np.sin(2.0 * np.pi * dow / 7.0)
        df['dow_cos'] = np.cos(2.0 * np.pi * dow / 7.0)

    # -------------------------------------------------------------------------
    # MULTI-TIMEFRAME FEATURES (MTF)
    # -------------------------------------------------------------------------
    if req is None or any(c in req for c in ('return_15m_agg', 'return_1h_agg', 'volatility_1h', 'rsi_15m', 'adx_14', 'efficiency_ratio_30', 'atr_14_norm')):
        if req is None or 'return_15m_agg' in req:
            df['return_15m_agg'] = df['close'].pct_change(15)
        if req is None or 'return_1h_agg' in req:
            df['return_1h_agg'] = df['close'].pct_change(60)
        if req is None or 'volatility_1h' in req:
            df['volatility_1h'] = df['return_1'].rolling(60).std()

        if req is None or 'rsi_15m' in req:
            gain15 = delta.where(delta > 0, 0.0).rolling(15).mean()
            loss15 = (-delta.where(delta < 0, 0.0).rolling(15).mean()).clip(lower=1e-10)
            rs15 = gain15 / loss15
            df['rsi_15m'] = 100.0 - (100.0 / (1.0 + rs15))

        atr14 = None
        if req is None or any(a in req for a in ('adx_14', 'atr_14_norm')):
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift(1)).abs(),
                (df['low'] - df['close'].shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean().clip(lower=1e-10)

        if req is None or 'adx_14' in req:
            plus_dm = (df['high'].diff()).clip(lower=0.0)
            minus_dm = (-df['low'].diff()).clip(lower=0.0)
            plus_di = 100.0 * plus_dm.rolling(14).mean() / atr14
            minus_di = 100.0 * minus_dm.rolling(14).mean() / atr14
            dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100.0
            df['adx_14'] = dx.rolling(14).mean()

        if req is None or 'efficiency_ratio_30' in req:
            net_change = (df['close'] - df['close'].shift(30)).abs()
            path_sum = df['close'].diff().abs().rolling(30).sum().clip(lower=1e-10)
            df['efficiency_ratio_30'] = net_change / path_sum

        if (req is None or 'atr_14_norm' in req) and atr14 is not None:
            df['atr_14_norm'] = atr14 / df['close'].clip(lower=1e-10)

    # Vectorized vol_percentile_240 (parity with root feature_engineering.py)
    if (req is None or 'vol_percentile_240' in req) and 'volatility_14' in df.columns:
        v14 = df['volatility_14'].to_numpy(dtype=np.float64)
        if len(v14) >= 240:
            sub_windows = np.lib.stride_tricks.sliding_window_view(v14, 240)
            has_nan = np.isnan(sub_windows).any(axis=1)
            last = sub_windows[:, -1:]
            cmp = (sub_windows >= last) if legacy_vol_percentile else (sub_windows <= last)
            pct = cmp.mean(axis=1) * 100.0
            pct[has_nan] = np.nan
            out = np.full(len(v14), np.nan, dtype=np.float64)
            out[239:] = pct
            df['vol_percentile_240'] = out
        else:
            df['vol_percentile_240'] = np.nan

    # -------------------------------------------------------------------------
    # FE 3.0 CANDLE & MICROSTRUCTURE FEATURES
    # -------------------------------------------------------------------------
    candle_feats = ('candle_body_ratio', 'close_position', 'upper_shadow')
    if req is None or any(c in req for c in candle_feats):
        hl_diff = (df['high'] - df['low']).clip(lower=1e-10)
        if req is None or 'candle_body_ratio' in req:
            df['candle_body_ratio'] = (df['close'] - df['open']).abs() / hl_diff
        if req is None or 'close_position' in req:
            df['close_position'] = (df['close'] - df['low']) / hl_diff
        if req is None or 'upper_shadow' in req:
            df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / hl_diff

    if req is None or 'momentum_accel' in req:
        df['momentum_accel'] = df['return_1'].diff(5)

    if req is None or 'volume_trend' in req:
        vol_sma_14 = df['volume'].rolling(14).mean()
        vol_sma_60 = df['volume'].rolling(60).mean().clip(lower=1e-10)
        df['volume_trend'] = vol_sma_14 / vol_sma_60

    if req is None or 'vwap_position' in req:
        vwap_proxy = (
            (df['close'] * df['volume']).rolling(20).sum()
            / df['volume'].rolling(20).sum().clip(lower=1e-10)
        )
        df['vwap_position'] = df['close'] / vwap_proxy.clip(lower=1e-10)

    if req is None or 'dist_to_threshold_sigma' in req:
        rolling_sigma = df['return_1'].rolling(20).std().clip(lower=1e-10)
        df['dist_to_threshold_sigma'] = df['return_1'] / rolling_sigma

    return df


def get_feature_columns(include_buy_pressure: bool = False,
                        include_mtf: bool = False) -> list[str]:
    """Canonical ordered list of generated feature names."""
    features = [
        'return_1', 'return_5', 'return_15', 'return_60',
        'close_sma_7', 'close_sma_14', 'close_sma_30',
        'macd', 'macd_signal', 'macd_diff', 'rsi',
        'volatility_7', 'volatility_14', 'volatility_30',
        'vol_regime', 'relative_volume',
    ]
    if include_buy_pressure:
        features.append('buy_pressure')
    features.extend(['bb_position', 'hl_range', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'])

    if include_mtf:
        features.append('volatility_3')
        features.extend([
            'return_15m_agg', 'return_1h_agg', 'volatility_1h', 'rsi_15m',
            'adx_14', 'efficiency_ratio_30', 'vol_percentile_240',
            'atr_14_norm', 'candle_body_ratio', 'close_position', 'upper_shadow',
            'momentum_accel', 'volume_trend', 'vwap_position', 'dist_to_threshold_sigma',
        ])
    return features
