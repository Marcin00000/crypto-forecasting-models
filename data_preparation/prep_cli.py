#!/usr/bin/env python
"""
CLI Interface for Data Preparation
"""
import argparse
from prepare_data import DataPreparation
from model_adapters.feature_engineering import get_feature_sets


def main():
    parser = argparse.ArgumentParser(description='Prepare cryptocurrency data for model training')

    parser.add_argument('data_path', type=str, help='Path to raw CSV data file')

    parser.add_argument('--models', type=str, nargs='+', default=['lstm'],
                        choices=['lstm', 'xgb', 'universal'],
                        help='Models to prepare data for (default: lstm)')
    parser.add_argument('--data_types', type=str, nargs='+',
                        default=['train', 'test'],
                        choices=['train', 'agent', 'test', 'sym'],
                        help='Data types to include (default: train test)')
    parser.add_argument('--output_format', type=str, default='native',
                        choices=['native', 'csv', 'both'],
                        help='Output format (default: native)')
    parser.add_argument('--output_dir', type=str, default='prepared_data',
                        help='Output directory (default: prepared_data)')

    parser.add_argument('--split_mode', type=str, default='months',
                        choices=['months', 'percentage', 'dates'],
                        help='Data split mode (default: months)')

    parser.add_argument('--train_months', type=int, default=None)
    parser.add_argument('--agent_months', type=int, default=None)
    parser.add_argument('--test_months', type=int, default=None)
    parser.add_argument('--sym_months', type=int, default=None)

    parser.add_argument('--train_pct', type=float, default=None)
    parser.add_argument('--agent_pct', type=float, default=None)
    parser.add_argument('--test_pct', type=float, default=None)
    parser.add_argument('--sym_pct', type=float, default=None)

    parser.add_argument('--train_start', type=str, default=None)
    parser.add_argument('--train_end', type=str, default=None)
    parser.add_argument('--agent_start', type=str, default=None)
    parser.add_argument('--agent_end', type=str, default=None)
    parser.add_argument('--test_start', type=str, default=None)
    parser.add_argument('--test_end', type=str, default=None)
    parser.add_argument('--sym_start', type=str, default=None)
    parser.add_argument('--sym_end', type=str, default=None)

    parser.add_argument('--feature_set', type=str, default='full',
                        choices=sorted(get_feature_sets().keys()),
                        help='Feature set (default: full)')
    parser.add_argument('--custom_features', type=str, nargs='+', default=None,
                        help='Custom feature list')
    parser.add_argument('--lookback', type=int, default=48,
                        help='Lookback window (default: 48)')
    parser.add_argument('--horizon', type=int, default=12,
                        help='Prediction horizon (default: 12, as in datasets E109-E115)')
    parser.add_argument('--data_id', type=str, default=None,
                        help='Pin the package ID (e.g. E116) instead of taking the next free one')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow reusing the package directory of an already existing --data_id')

    parser.add_argument('--scaler_features', type=str, default='standard',
                        choices=['minmax', 'standard', 'robust', 'none', 'raw', 'quantile_normal', 'quantile_uniform'],
                        help='Feature scaler (default: standard)')
    parser.add_argument('--scaler_price', type=str, default='standard',
                        choices=['minmax', 'standard', 'robust', 'none', 'raw', 'quantile_normal', 'quantile_uniform'],
                        help='Price scaler for LSTM (default: standard)')
    parser.add_argument('--no_log_prices', action='store_true',
                        help='Disable log prices for LSTM')
    parser.add_argument('--no_scale_csv', action='store_true',
                        help='Disable feature scaling in CSV files (keep raw features)')

    args = parser.parse_args()

    if 'universal' in args.models:
        model_types = 'universal'
    else:
        model_types = args.models

    if args.split_mode == 'months':
        if args.train_months is None:
            args.train_months = 3
        if args.test_months is None:
            args.test_months = 1
    elif args.split_mode == 'percentage':
        if args.train_pct is None:
            args.train_pct = 70.0
        if args.test_pct is None:
            args.test_pct = 30.0

    try:
        prep = DataPreparation(output_dir=args.output_dir)

        output_path, data_id = prep.prepare(
            data_path=args.data_path,
            model_types=model_types,
            data_types=args.data_types,
            output_format=args.output_format,
            split_mode=args.split_mode,
            train_months=args.train_months,
            train_pct=args.train_pct,
            train_start=args.train_start,
            train_end=args.train_end,
            agent_months=args.agent_months,
            agent_pct=args.agent_pct,
            agent_start=args.agent_start,
            agent_end=args.agent_end,
            test_months=args.test_months,
            test_pct=args.test_pct,
            test_start=args.test_start,
            test_end=args.test_end,
            sym_months=args.sym_months,
            sym_pct=args.sym_pct,
            sym_start=args.sym_start,
            sym_end=args.sym_end,
            feature_set=args.feature_set,
            custom_features=args.custom_features,
            lookback=args.lookback,
            horizon=args.horizon,
            scaler_features=args.scaler_features,
            scaler_price=args.scaler_price,
            use_log_prices=not args.no_log_prices,
            scale_csv=not args.no_scale_csv,
            data_id=args.data_id,
            overwrite=args.overwrite,
        )

        print(f"\nSuccess! Data ID: {data_id}")
        print(f"Package: {output_path}")

    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR")
        print("=" * 70)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        raise


if __name__ == "__main__":
    main()
