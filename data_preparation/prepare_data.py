"""
Main Data Preparation Module
"""
import os
import json
import pandas as pd
import numpy as np

from prep_config import PrepConfig
from prep_utils import (detect_symbol, detect_interval, clean_data,
                        split_by_months, split_by_percentage, split_by_dates)
from prep_id import resolve_data_id, generate_package_name, create_metadata


def validate_data_split(train_df, agent_df, test_df, sym_df, date_ranges,
                        train_months, agent_months, test_months, sym_months):
    """Validate that data split is correct"""
    print("\n   === VALIDATION ===")

    splits = []
    if train_df is not None and len(train_df) > 0:
        splits.append(('train', date_ranges['train']))
    if agent_df is not None and len(agent_df) > 0:
        splits.append(('agent', date_ranges['agent']))
    if test_df is not None and len(test_df) > 0:
        splits.append(('test', date_ranges['test']))
    if sym_df is not None and len(sym_df) > 0:
        splits.append(('sym', date_ranges['sym']))

    for i in range(len(splits) - 1):
        name1, (start1, end1) = splits[i]
        name2, (start2, end2) = splits[i + 1]

        if end1 is None or start2 is None:
            continue

        if pd.to_datetime(end1) >= pd.to_datetime(start2):
            raise ValueError(
                f"❌ Overlap: {name1} ends at {end1}, {name2} starts at {start2}"
            )

    print("   ✓ Chronological order: TRAIN → AGENT → TEST → SYM")

    def check_duration(name, date_range, expected_months):
        if date_range is None or date_range[0] is None:
            return
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        actual_days = (end - start).days
        expected_days = expected_months * 30
        tolerance = 2

        if abs(actual_days - expected_days) > tolerance:
            print(f"   ⚠️  {name.upper()}: {actual_days} days (expected ~{expected_days})")
        else:
            print(f"   ✓ {name.upper()}: {actual_days} days (~{expected_months} months)")

    if train_df is not None and len(train_df) > 0:
        check_duration('train', date_ranges['train'], train_months)
    if agent_df is not None and len(agent_df) > 0:
        check_duration('agent', date_ranges['agent'], agent_months)
    if test_df is not None and len(test_df) > 0:
        check_duration('test', date_ranges['test'], test_months)
    if sym_df is not None and len(sym_df) > 0:
        check_duration('sym', date_ranges['sym'], sym_months)

    print("   === VALIDATION PASSED ===\n")


class DataPreparation:
    """Main data preparation orchestrator"""

    def __init__(self, output_dir=None):
        self.output_dir = output_dir or PrepConfig.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_csv_split(self, df, features, output_path, filename, lookback,
                        scale_csv, scaler, include_buy_pressure=False):
        """
        Save a CSV split with feature engineering applied.

        Calls prepare_features() from the universal FE_ (via model_adapters
        bridge) so that new features (vol_regime, buy_pressure, bugfixes) are
        present in every saved CSV.

        Args:
            include_buy_pressure: Must match what was used when selecting
                features — ensures 'buy_pressure' column exists before the
                scaler tries to transform it.
        """
        from model_adapters.feature_engineering import prepare_features

        df_csv = df.copy()
        df_csv = prepare_features(df_csv)
        df_csv = df_csv.replace([np.inf, -np.inf], np.nan).dropna()

        # Remove 2*lookback rows to ensure no cross-contamination between splits
        df_csv = df_csv.iloc[2 * lookback:].reset_index(drop=True)

        if scale_csv and scaler is not None:
            df_csv[features] = scaler.transform(df_csv[features])

        # Base OHLCV columns to prepend in the CSV
        base_ohlcv = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        if 'close_time' in df_csv.columns:
            base_ohlcv.insert(1, 'close_time')
        available_ohlcv = [col for col in base_ohlcv
                           if col in df_csv.columns and col not in features]
        cols_to_save = available_ohlcv + features

        df_csv[cols_to_save].to_csv(os.path.join(output_path, filename), index=False)
        num_rows = len(df_csv)
        print(f"   Saved {filename}: {num_rows} rows "
              f"(removed {2 * lookback} buffer, scaled={scale_csv})")
        return num_rows

    def prepare(self,
                data_path,
                model_types=None,
                data_types=None,
                output_format='native',
                split_mode='months',
                train_months=None, train_pct=None, train_start=None, train_end=None,
                agent_months=None, agent_pct=None, agent_start=None, agent_end=None,
                test_months=None, test_pct=None, test_start=None, test_end=None,
                sym_months=None, sym_pct=None, sym_start=None, sym_end=None,
                feature_set='full',
                custom_features=None,
                lookback=48,
                horizon=1,
                scaler_features='standard',
                scaler_price='standard',
                use_log_prices=True,
                scale_csv=True,
                data_id=None,
                overwrite=False,
                ):
        """Prepare data for model training.

        data_id pins the package ID (e.g. 'E114') so re-running a generator script
        cannot mint the next free ID and desynchronise the dataset registries;
        an existing package is only reused with overwrite=True.
        """
        print("=" * 70)
        print("DATA PREPARATION")
        print("=" * 70)

        if model_types is None:
            model_types = ['lstm']
        elif model_types == 'universal':
            model_types = PrepConfig.SUPPORTED_MODELS

        if data_types is None:
            data_types = [PrepConfig.DATA_TYPE_TRAIN, PrepConfig.DATA_TYPE_TEST]
            if split_mode == 'months':
                if agent_months and agent_months > 0:
                    data_types.append(PrepConfig.DATA_TYPE_AGENT)
                if sym_months and sym_months > 0:
                    data_types.append(PrepConfig.DATA_TYPE_SYM)
            elif split_mode == 'percentage':
                if agent_pct and agent_pct > 0:
                    data_types.append(PrepConfig.DATA_TYPE_AGENT)
                if sym_pct and sym_pct > 0:
                    data_types.append(PrepConfig.DATA_TYPE_SYM)
            elif split_mode == 'dates':
                if agent_start and agent_end:
                    data_types.append(PrepConfig.DATA_TYPE_AGENT)
                if sym_start and sym_end:
                    data_types.append(PrepConfig.DATA_TYPE_SYM)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        for model_type in model_types:
            if model_type not in PrepConfig.SUPPORTED_MODELS:
                raise ValueError(f"Unsupported model: {model_type}")

        # ------------------------------------------------------------------
        # 1. Load raw data
        # ------------------------------------------------------------------
        print(f"\n1. Loading data: {data_path}")
        try:
            if data_path.endswith('.npy'):
                data = np.load(data_path, allow_pickle=True)
                df = pd.DataFrame(data)
            elif data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            raise ValueError(f"Failed to load data: {e}")

        print(f"   Loaded {len(df)} records")

        symbol = detect_symbol(os.path.basename(data_path))
        interval = detect_interval(df)
        print(f"   Symbol: {symbol.upper()}, Interval: {interval}")

        # ------------------------------------------------------------------
        # 2. Clean — keeps taker_buy_base_asset_volume when present
        # ------------------------------------------------------------------
        print("\n2. Cleaning data")
        df, removed = clean_data(df, min_required_rows=72)
        print(f"   Removed {removed} rows, remaining: {len(df)}")

        # Detect whether buy_pressure can be computed for this dataset
        has_taker_buy = 'taker_buy_base_asset_volume' in df.columns
        if has_taker_buy:
            print("   [OK] taker_buy_base_asset_volume present -> buy_pressure will be computed")
        else:
            print("   [INFO] taker_buy_base_asset_volume absent -> buy_pressure skipped")

        if not pd.api.types.is_datetime64_any_dtype(df['open_time']):
            try:
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            except Exception:
                df['open_time'] = pd.to_datetime(df['open_time'])

        from prep_utils import parse_datetime
        df_temp = parse_datetime(df.copy())
        raw_date_start = df_temp['datetime'].min()
        raw_date_end = df_temp['datetime'].max()
        print(f"   Date range: {raw_date_start} to {raw_date_end}")

        # ------------------------------------------------------------------
        # 3. Split
        # ------------------------------------------------------------------
        print(f"\n3. Splitting data (mode: {split_mode})")
        split_config = {'mode': split_mode, 'params': {}}

        if split_mode == 'months':
            if train_months is None or test_months is None:
                raise ValueError("Must provide train_months and test_months")
            split_config['params'] = {
                'train_months': train_months,
                'agent_months': agent_months or 0,
                'test_months': test_months,
                'sym_months': sym_months or 0,
            }
            train_df, agent_df, test_df, sym_df, date_ranges = split_by_months(
                df, train_months, agent_months or 0, test_months, sym_months or 0
            )
            validate_data_split(train_df, agent_df, test_df, sym_df, date_ranges,
                                 train_months, agent_months or 0, test_months, sym_months or 0)

        elif split_mode == 'percentage':
            if train_pct is None or test_pct is None:
                raise ValueError("Must provide train_pct and test_pct")
            split_config['params'] = {
                'train_pct': train_pct,
                'agent_pct': agent_pct or 0,
                'test_pct': test_pct,
                'sym_pct': sym_pct or 0,
            }
            train_df, agent_df, test_df, sym_df, date_ranges = split_by_percentage(
                df, train_pct, agent_pct or 0, test_pct, sym_pct or 0
            )

        elif split_mode == 'dates':
            if not train_start or not train_end or not test_start or not test_end:
                raise ValueError("Must provide train_start, train_end, test_start, test_end")
            split_config['params'] = {
                'train_start': str(train_start), 'train_end': str(train_end),
                'agent_start': str(agent_start) if agent_start else None,
                'agent_end': str(agent_end) if agent_end else None,
                'test_start': str(test_start), 'test_end': str(test_end),
                'sym_start': str(sym_start) if sym_start else None,
                'sym_end': str(sym_end) if sym_end else None,
            }
            train_df, agent_df, test_df, sym_df, date_ranges = split_by_dates(
                df, train_start, train_end, agent_start, agent_end,
                test_start, test_end, sym_start, sym_end
            )
        else:
            raise ValueError(f"Invalid split_mode: {split_mode}")

        if test_df is None or len(test_df) == 0:
            raise ValueError("Test data is empty after split")
        if train_df is None or len(train_df) == 0:
            raise ValueError("Train data is empty after split")

        min_samples = lookback + horizon + 10
        if len(train_df) < min_samples:
            raise ValueError(f"Insufficient training data: {len(train_df)} rows, need at least {min_samples}")
        if len(test_df) < min_samples:
            raise ValueError(f"Insufficient test data: {len(test_df)} rows, need at least {min_samples}")

        print(f"\n   Split results:")
        print(f"   Train: {len(train_df)} records ({date_ranges['train'][0]} to {date_ranges['train'][1]})")
        if agent_df is not None and len(agent_df) > 0:
            print(f"   Agent: {len(agent_df)} records ({date_ranges['agent'][0]} to {date_ranges['agent'][1]})")
        print(f"   Test:  {len(test_df)} records ({date_ranges['test'][0]} to {date_ranges['test'][1]})")
        if sym_df is not None and len(sym_df) > 0:
            print(f"   Sym:   {len(sym_df)} records ({date_ranges['sym'][0]} to {date_ranges['sym'][1]})")

        if len(model_types) == 1:
            model_type = model_types[0]
        else:
            model_type = None

        prefix = PrepConfig.get_symbol_prefix(symbol)
        data_id = resolve_data_id(self.output_dir, prefix, data_id, overwrite)
        package_name = generate_package_name(data_id, symbol, feature_set, interval, model_type)
        output_path = os.path.join(self.output_dir, package_name)
        os.makedirs(output_path, exist_ok=True)

        print(f"\n4. Output: {output_path}")
        print(f"   Data ID: {data_id}")
        print(f"   Data types: {', '.join(data_types)}")

        all_data = {}
        all_base_features = {}
        universal_feature_scaler = None
        universal_price_scaler = None

        # ------------------------------------------------------------------
        # 5. Feature engineering per model
        # ------------------------------------------------------------------
        for mt in model_types:
            print(f"\n5. Preparing features for {mt.upper()}")

            if mt == 'lstm':
                from model_adapters.lstm_adapter import (
                    prepare_features, select_features,
                    prepare_for_lstm, save_lstm_data,
                )

                train_df_copy = train_df.copy()
                agent_df_copy = agent_df.copy() if agent_df is not None else None
                test_df_copy = test_df.copy()
                sym_df_copy = sym_df.copy() if sym_df is not None else None

                train_df_copy = prepare_features(train_df_copy)
                test_df_copy = prepare_features(test_df_copy)
                if agent_df_copy is not None:
                    agent_df_copy = prepare_features(agent_df_copy)
                if sym_df_copy is not None:
                    sym_df_copy = prepare_features(sym_df_copy)

                features = select_features(
                    train_df_copy, feature_set, custom_features,
                    include_buy_pressure=has_taker_buy,
                )
                features = [f for f in features if f != 'close']

                if len(features) == 0:
                    raise ValueError("No features selected for LSTM")

                train_df_copy = train_df_copy.replace([np.inf, -np.inf], np.nan).dropna()
                test_df_copy = test_df_copy.replace([np.inf, -np.inf], np.nan).dropna()
                if agent_df_copy is not None:
                    agent_df_copy = agent_df_copy.replace([np.inf, -np.inf], np.nan).dropna()
                if sym_df_copy is not None:
                    sym_df_copy = sym_df_copy.replace([np.inf, -np.inf], np.nan).dropna()

                data = prepare_for_lstm(
                    train_df_copy, agent_df_copy, test_df_copy, sym_df_copy,
                    features, lookback, horizon,
                    scaler_features, scaler_price, use_log_prices,
                )

                all_data['lstm'] = data
                all_base_features['lstm'] = features

                universal_feature_scaler = data['feature_scaler']
                universal_price_scaler = data['price_scaler']

                if output_format in ['native', 'both']:
                    save_lstm_data(output_path, data, prefix='')
                    print("   Saved LSTM native format (NPY)")

            elif mt == 'xgb':
                from model_adapters.xgb_adapter import (
                    prepare_features, select_features,
                    prepare_for_xgb, )

                train_df_copy = train_df.copy()
                agent_df_copy = agent_df.copy() if agent_df is not None else None
                test_df_copy = test_df.copy()
                sym_df_copy = sym_df.copy() if sym_df is not None else None

                train_df_copy = prepare_features(train_df_copy)
                test_df_copy = prepare_features(test_df_copy)
                if agent_df_copy is not None:
                    agent_df_copy = prepare_features(agent_df_copy)
                if sym_df_copy is not None:
                    sym_df_copy = prepare_features(sym_df_copy)

                features = select_features(
                    train_df_copy, feature_set, custom_features,
                    include_buy_pressure=has_taker_buy,
                )
                features = [f for f in features if f != 'close']

                if len(features) == 0:
                    raise ValueError("No features selected for XGBoost")

                train_df_copy = train_df_copy.replace([np.inf, -np.inf], np.nan).dropna()
                test_df_copy = test_df_copy.replace([np.inf, -np.inf], np.nan).dropna()
                if agent_df_copy is not None:
                    agent_df_copy = agent_df_copy.replace([np.inf, -np.inf], np.nan).dropna()
                if sym_df_copy is not None:
                    sym_df_copy = sym_df_copy.replace([np.inf, -np.inf], np.nan).dropna()

                data = prepare_for_xgb(
                    train_df_copy, agent_df_copy, test_df_copy, sym_df_copy,
                    features, lookback, horizon,
                    scaler_features,
                )

                all_data['xgb'] = data
                all_base_features['xgb'] = features

                if universal_feature_scaler is None:
                    universal_feature_scaler = data['feature_scaler']

                print("   XGBoost will use CSV files (no NPY)")

        # ------------------------------------------------------------------
        # 6. Save CSV files
        # ------------------------------------------------------------------
        csv_actual_rows = {}
        should_save_csv = (
            'xgb' in model_types
            or output_format in ['csv', 'both']
            or (PrepConfig.DATA_TYPE_AGENT in data_types and agent_df is not None)
            or (PrepConfig.DATA_TYPE_SYM in data_types and sym_df is not None)
        )

        if should_save_csv:
            print(f"\n6. Saving CSV files (OHLCV + features)")
            if 'xgb' in model_types:
                print("   Note: CSV required for XGBoost")
            if PrepConfig.DATA_TYPE_AGENT in data_types or PrepConfig.DATA_TYPE_SYM in data_types:
                print("   Note: CSV required for agent/sym data")

            features_to_save = all_base_features[model_types[0]]

            csv_kwargs = dict(
                lookback=lookback,
                scale_csv=scale_csv,
                scaler=universal_feature_scaler,
                include_buy_pressure=has_taker_buy,
            )

            if PrepConfig.DATA_TYPE_TRAIN in data_types:
                csv_actual_rows['train'] = self._save_csv_split(
                    train_df, features_to_save, output_path, 'train.csv', **csv_kwargs)

            if PrepConfig.DATA_TYPE_TEST in data_types:
                csv_actual_rows['test'] = self._save_csv_split(
                    test_df, features_to_save, output_path, 'test.csv', **csv_kwargs)

            if PrepConfig.DATA_TYPE_AGENT in data_types and agent_df is not None and len(agent_df) > 0:
                csv_actual_rows['agent'] = self._save_csv_split(
                    agent_df, features_to_save, output_path, 'agent.csv', **csv_kwargs)

            if PrepConfig.DATA_TYPE_SYM in data_types and sym_df is not None and len(sym_df) > 0:
                csv_actual_rows['sym'] = self._save_csv_split(
                    sym_df, features_to_save, output_path, 'sym.csv', **csv_kwargs)

        # ------------------------------------------------------------------
        # Save scalers
        # ------------------------------------------------------------------
        import joblib
        joblib.dump(universal_feature_scaler, os.path.join(output_path, 'scaler_features.pkl'))
        if universal_price_scaler is not None:
            joblib.dump(universal_price_scaler, os.path.join(output_path, 'scaler_price.pkl'))
        print(f"\n   Saved scalers: scaler_features.pkl"
              + (", scaler_price.pkl" if universal_price_scaler else ""))

        # ------------------------------------------------------------------
        # 7. Metadata
        # ------------------------------------------------------------------
        print("\n7. Saving metadata")

        data_shapes = {}
        data_counts = {}

        for mt in model_types:
            if mt == 'lstm':
                d = all_data['lstm']
                data_shapes['lstm'] = {
                    'X_train': list(d['X_train'].shape),
                    'y_train': list(d['y_train'].shape),
                    'X_test':  list(d['X_test'].shape),
                    'y_test':  list(d['y_test'].shape),
                    'test_2d': list(d['test_2d'].shape),
                }
                data_counts['lstm'] = {
                    'train_sequences': int(d['X_train'].shape[0]),
                    'test_sequences':  int(d['X_test'].shape[0]),
                    'train_raw_records': len(train_df),
                    'test_raw_records':  len(test_df),
                }
                if d.get('agent_data') is not None:
                    data_shapes['lstm']['agent_data'] = list(d['agent_data'].shape)
                    data_counts['lstm']['agent_raw_records'] = len(agent_df)
                if d.get('sym_data') is not None:
                    data_shapes['lstm']['sym_data'] = list(d['sym_data'].shape)
                    data_counts['lstm']['sym_raw_records'] = len(sym_df)

            elif mt == 'xgb':
                data_shapes['xgb'] = 'csv_based'
                data_counts['xgb'] = {
                    'train_raw_records': len(train_df),
                    'test_raw_records':  len(test_df),
                }
                if 'train' in csv_actual_rows:
                    data_counts['xgb']['train_csv_rows'] = csv_actual_rows['train']
                if 'test' in csv_actual_rows:
                    data_counts['xgb']['test_csv_rows'] = csv_actual_rows['test']
                if agent_df is not None and len(agent_df) > 0:
                    data_counts['xgb']['agent_raw_records'] = len(agent_df)
                    if 'agent' in csv_actual_rows:
                        data_counts['xgb']['agent_csv_rows'] = csv_actual_rows['agent']
                if sym_df is not None and len(sym_df) > 0:
                    data_counts['xgb']['sym_raw_records'] = len(sym_df)
                    if 'sym' in csv_actual_rows:
                        data_counts['xgb']['sym_csv_rows'] = csv_actual_rows['sym']

        scalers_info = {
            'features': scaler_features,
            'price': scaler_price,
            'use_log_prices': use_log_prices,
            'scale_csv': scale_csv,
        }

        features_for_metadata = all_base_features[model_types[0]]

        metadata = create_metadata(
            data_id, symbol, interval, feature_set, features_for_metadata,
            lookback, horizon, split_config, date_ranges,
            data_shapes, scalers_info, model_type, data_counts,
        )
        # Record buy_pressure availability in metadata
        metadata['has_buy_pressure'] = has_taker_buy

        metadata_path = os.path.join(output_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print("   Saved metadata.json")

        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("PREPARATION COMPLETE")
        print("=" * 70)
        print(f"Data ID:  {data_id}")
        print(f"Package:  {output_path}")
        print(f"Models:   {', '.join([m.upper() for m in model_types])}")
        print(f"Data types: {', '.join(data_types)}")
        if 'lstm' in model_types and output_format in ['native', 'both']:
            print("LSTM:     NPY sequences saved")
        if 'xgb' in model_types or should_save_csv:
            csv_files = "train.csv, test.csv"
            if agent_df is not None and len(agent_df) > 0:
                csv_files += ", agent.csv"
            if sym_df is not None and len(sym_df) > 0:
                csv_files += ", sym.csv"
            print(f"CSV:      {csv_files}")
        print(f"Scalers:  scaler_features.pkl"
              + (", scaler_price.pkl" if universal_price_scaler else ""))
        if has_taker_buy:
            print("Features: buy_pressure included (taker_buy column present)")
        print("=" * 70)

        return output_path, data_id
