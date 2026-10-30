"""
Data splitting and cleaning utilities
"""
import pandas as pd
import numpy as np


def detect_symbol(filename):
    """Detect cryptocurrency symbol from filename"""
    filename_lower = filename.lower()
    symbols = ['btc', 'eth', 'bnb', 'sol', 'ada', 'xrp', 'doge', 'dot', 'matic', 'avax']
    for sym in symbols:
        if sym in filename_lower:
            return sym
    return 'crypto'


def detect_interval(df):
    """Detect time interval from dataframe"""
    df_temp = df.copy()
    try:
        df_temp['datetime'] = pd.to_datetime(df_temp['open_time'], unit='ms', errors='coerce')
        if df_temp['datetime'].isna().all():
            raise ValueError("Not Unix timestamps")
    except (ValueError, TypeError):
        df_temp['datetime'] = pd.to_datetime(df_temp['open_time'])

    time_diffs = df_temp['datetime'].diff().dropna()
    if len(time_diffs) == 0:
        return 'unknown'

    median_diff = time_diffs.median().total_seconds()

    interval_map = {
        60: '1m', 300: '5m', 900: '15m', 1800: '30m',
        3600: '1h', 14400: '4h', 86400: '1d'
    }

    return interval_map.get(int(median_diff), f"{int(median_diff)}s")


def parse_datetime(df):
    """Parse datetime column with fallback"""
    try:
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
        if df['datetime'].isna().all():
            raise ValueError("Not Unix timestamps")
    except (ValueError, TypeError):
        df['datetime'] = pd.to_datetime(df['open_time'])
    return df


def split_by_months(df, train_months, agent_months, test_months, sym_months=0):
    """
    Split data by month lengths in chronological order: TRAIN → AGENT → TEST → SYM

    Each split is non-overlapping and chronologically ordered from oldest to newest.
    """
    if 'open_time' not in df.columns:
        raise ValueError("Missing 'open_time' column")

    df_copy = parse_datetime(df)
    df_copy = df_copy.sort_values('datetime').reset_index(drop=True)

    latest_date = df_copy['datetime'].max()
    earliest_date = df_copy['datetime'].min()

    # Calculate boundaries from newest to oldest
    sym_end = latest_date
    sym_start = sym_end - pd.Timedelta(days=sym_months * 30) if sym_months > 0 else None

    test_end = sym_start if sym_months > 0 else latest_date
    test_start = test_end - pd.Timedelta(days=test_months * 30)

    agent_end = test_start
    agent_start = agent_end - pd.Timedelta(days=agent_months * 30) if agent_months > 0 else None

    train_end = agent_start if agent_months > 0 else test_start
    train_start = train_end - pd.Timedelta(days=train_months * 30)

    if train_start < earliest_date:
        raise ValueError(
            f"Insufficient data: need {train_months + agent_months + test_months + sym_months} months, "
            f"but dataset spans {(latest_date - earliest_date).days / 30:.1f} months"
        )

    train_df = df_copy[(df_copy['datetime'] >= train_start) & (df_copy['datetime'] < train_end)].copy()
    test_df = df_copy[(df_copy['datetime'] >= test_start) & (df_copy['datetime'] < test_end)].copy()

    agent_df = None
    if agent_months > 0:
        agent_df = df_copy[(df_copy['datetime'] >= agent_start) & (df_copy['datetime'] < agent_end)].copy()

    sym_df = None
    if sym_months > 0:
        sym_df = df_copy[(df_copy['datetime'] >= sym_start) & (df_copy['datetime'] <= sym_end)].copy()

    def verify_no_overlap(name1, df1, name2, df2):
        if df1 is None or df2 is None or len(df1) == 0 or len(df2) == 0:
            return
        max1 = df1['datetime'].max()
        min2 = df2['datetime'].min()
        if max1 >= min2:
            raise ValueError(f"Data overlap detected: {name1} ends at {max1}, {name2} starts at {min2}")

    verify_no_overlap('train', train_df, 'agent', agent_df)
    verify_no_overlap('agent', agent_df, 'test', test_df)
    verify_no_overlap('test', test_df, 'sym', sym_df)
    if agent_df is None:
        verify_no_overlap('train', train_df, 'test', test_df)

    date_ranges = {
        'train': (train_df['datetime'].min(), train_df['datetime'].max()) if len(train_df) > 0 else (None, None),
        'agent': (agent_df['datetime'].min(), agent_df['datetime'].max()) if agent_df is not None and len(agent_df) > 0 else None,
        'test': (test_df['datetime'].min(), test_df['datetime'].max()) if len(test_df) > 0 else (None, None),
        'sym': (sym_df['datetime'].min(), sym_df['datetime'].max()) if sym_df is not None and len(sym_df) > 0 else None,
    }

    for df_item in [train_df, test_df]:
        if 'datetime' in df_item.columns:
            df_item.drop('datetime', axis=1, inplace=True)
    if agent_df is not None and 'datetime' in agent_df.columns:
        agent_df.drop('datetime', axis=1, inplace=True)
    if sym_df is not None and 'datetime' in sym_df.columns:
        sym_df.drop('datetime', axis=1, inplace=True)

    return train_df, agent_df, test_df, sym_df, date_ranges


def split_by_percentage(df, train_pct, agent_pct, test_pct, sym_pct=0):
    """
    Split data by percentages in chronological order: TRAIN → AGENT → TEST → SYM
    """
    if 'open_time' not in df.columns:
        raise ValueError("Missing 'open_time' column")

    total_pct = train_pct + agent_pct + test_pct + sym_pct
    if abs(total_pct - 100.0) > 0.01:
        raise ValueError(f"Percentages must sum to 100, got {total_pct}")

    df_copy = parse_datetime(df)
    df_copy = df_copy.sort_values('datetime').reset_index(drop=True)

    n = len(df_copy)

    train_end_idx = int(n * train_pct / 100)
    agent_end_idx = train_end_idx + int(n * agent_pct / 100)
    test_end_idx = agent_end_idx + int(n * test_pct / 100)

    train_df = df_copy.iloc[:train_end_idx].copy()

    agent_df = None
    if agent_pct > 0:
        agent_df = df_copy.iloc[train_end_idx:agent_end_idx].copy()

    test_df = df_copy.iloc[agent_end_idx:test_end_idx].copy()

    sym_df = None
    if sym_pct > 0:
        sym_df = df_copy.iloc[test_end_idx:].copy()

    date_ranges = {
        'train': (train_df['datetime'].min(), train_df['datetime'].max()) if len(train_df) > 0 else (None, None),
        'agent': (agent_df['datetime'].min(), agent_df['datetime'].max()) if agent_df is not None and len(agent_df) > 0 else None,
        'test': (test_df['datetime'].min(), test_df['datetime'].max()) if len(test_df) > 0 else (None, None),
        'sym': (sym_df['datetime'].min(), sym_df['datetime'].max()) if sym_df is not None and len(sym_df) > 0 else None,
    }

    for df_item in [train_df, test_df]:
        if 'datetime' in df_item.columns:
            df_item.drop('datetime', axis=1, inplace=True)
    if agent_df is not None and 'datetime' in agent_df.columns:
        agent_df.drop('datetime', axis=1, inplace=True)
    if sym_df is not None and 'datetime' in sym_df.columns:
        sym_df.drop('datetime', axis=1, inplace=True)

    return train_df, agent_df, test_df, sym_df, date_ranges


def split_by_dates(df, train_start, train_end, agent_start=None, agent_end=None,
                   test_start=None, test_end=None, sym_start=None, sym_end=None):
    """Split data by explicit date ranges"""
    if 'open_time' not in df.columns:
        raise ValueError("Missing 'open_time' column")

    df_copy = parse_datetime(df)
    df_copy = df_copy.sort_values('datetime').reset_index(drop=True)

    def parse_date(date_str):
        if date_str is None:
            return None
        if isinstance(date_str, str):
            return pd.to_datetime(date_str)
        return date_str

    train_start = parse_date(train_start)
    train_end = parse_date(train_end)

    train_df = df_copy[(df_copy['datetime'] >= train_start) & (df_copy['datetime'] <= train_end)].copy()

    if agent_start and agent_end:
        agent_start = parse_date(agent_start)
        agent_end = parse_date(agent_end)
        agent_df = df_copy[(df_copy['datetime'] >= agent_start) & (df_copy['datetime'] <= agent_end)].copy()
    else:
        agent_df = None

    if test_start and test_end:
        test_start = parse_date(test_start)
        test_end = parse_date(test_end)
        test_df = df_copy[(df_copy['datetime'] >= test_start) & (df_copy['datetime'] <= test_end)].copy()
    else:
        raise ValueError("Must provide test_start and test_end for dates split mode")

    if sym_start and sym_end:
        sym_start = parse_date(sym_start)
        sym_end = parse_date(sym_end)
        sym_df = df_copy[(df_copy['datetime'] >= sym_start) & (df_copy['datetime'] <= sym_end)].copy()
    else:
        sym_df = None

    date_ranges = {
        'train': (train_df['datetime'].min(), train_df['datetime'].max()) if len(train_df) > 0 else (None, None),
        'agent': (agent_df['datetime'].min(), agent_df['datetime'].max()) if agent_df is not None and len(agent_df) > 0 else None,
        'test': (test_df['datetime'].min(), test_df['datetime'].max()) if test_df is not None and len(test_df) > 0 else (None, None),
        'sym': (sym_df['datetime'].min(), sym_df['datetime'].max()) if sym_df is not None and len(sym_df) > 0 else None,
    }

    for df_item in [train_df, test_df]:
        if df_item is not None and 'datetime' in df_item.columns:
            df_item.drop('datetime', axis=1, inplace=True)
    if agent_df is not None and 'datetime' in agent_df.columns:
        agent_df.drop('datetime', axis=1, inplace=True)
    if sym_df is not None and 'datetime' in sym_df.columns:
        sym_df.drop('datetime', axis=1, inplace=True)

    return train_df, agent_df, test_df, sym_df, date_ranges


def clean_data(df, min_required_rows=72):
    """
    Clean raw OHLCV data.

    Keeps the standard OHLCV columns plus taker_buy_base_asset_volume when
    present — feature_engineering.prepare_features() consumes it to produce
    the 'buy_pressure' feature.  All other Binance passthrough columns
    (close_time, qav, num_trades, taker_quote_vol, ignore) are dropped.
    """
    required_cols = ['open_time', 'open', 'high', 'low', 'close']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Base columns always kept
    keep_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']

    # Optional: close_time (kept for downstream metadata if present)
    if 'close_time' in df.columns:
        keep_cols.insert(1, 'close_time')

    # Optional: taker_buy_base_asset_volume → consumed by prepare_features()
    # to produce 'buy_pressure'.  Must survive clean_data so FE_ can see it.
    if 'taker_buy_base_asset_volume' in df.columns:
        keep_cols.append('taker_buy_base_asset_volume')

    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    initial_len = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    if len(df) < min_required_rows:
        raise ValueError(
            f"Insufficient data after cleaning: {len(df)} rows. "
            f"Need at least {min_required_rows} records."
        )

    removed = initial_len - len(df)

    return df, removed
