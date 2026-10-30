"""
ID management for prepared data packages
"""
import os
from datetime import datetime


def get_next_data_id(output_dir, prefix):
    """Get next available data ID with given prefix"""
    if not os.path.exists(output_dir):
        return f"{prefix}001"

    existing = [d for d in os.listdir(output_dir)
                if d.startswith(prefix) and '_prep_' in d]
    if not existing:
        return f"{prefix}001"

    numbers = []
    for d in existing:
        try:
            num = int(d[len(prefix):len(prefix)+3])
            numbers.append(num)
        except ValueError:
            continue

    next_num = max(numbers) + 1 if numbers else 1
    return f"{prefix}{next_num:03d}"


def resolve_data_id(output_dir, prefix, data_id=None, overwrite=False):
    """Return the requested data ID, or the next free one when none is given.

    A pinned ID must keep the symbol prefix and must not collide with an existing
    package unless overwrite is set.
    """
    if data_id is None:
        return get_next_data_id(output_dir, prefix)
    if not data_id.startswith(prefix):
        raise ValueError(f"data_id '{data_id}' must start with symbol prefix '{prefix}'")
    if os.path.isdir(output_dir):
        clash = [d for d in os.listdir(output_dir) if d.startswith(f"{data_id}_prep_")]
        if clash and not overwrite:
            raise FileExistsError(
                f"Package for data_id '{data_id}' already exists: {clash[0]}. "
                f"Pass overwrite=True to replace it.")
    return data_id


def generate_package_name(data_id, symbol, feature_set, interval, model_type=None):
    """
    Generate prepared data package name
    
    Args:
        data_id: Data ID based on symbol (e.g., 'B001', 'E001')
        symbol: Symbol (e.g., 'btc')
        feature_set: Feature set name
        interval: Time interval
        model_type: Model type ('lstm', 'xgb', or None for universal)
    
    Returns:
        str: Package directory name
    """
    if model_type:
        return f"{data_id}_prep_{symbol}_{model_type}_{feature_set}_{interval}"
    else:
        return f"{data_id}_prep_{symbol}_universal_{feature_set}_{interval}"


def create_metadata(data_id, symbol, interval, feature_set, features, 
                    lookback, horizon, split_config, date_ranges,
                    data_shapes, scalers_info, model_type=None, data_counts=None):
    """Create metadata dictionary for prepared data package"""
    metadata = {
        'data_id': data_id,
        'symbol': symbol,
        'interval': interval,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'feature_set': feature_set,
        'features': features,
        'n_features': len(features),
        'lookback': lookback,
        'horizon': horizon,
        'split_mode': split_config['mode'],
        'split_params': split_config['params'],
        'date_ranges': {
            k: {'start': str(v[0]) if v and v[0] else None,
                'end': str(v[1]) if v and v[1] else None}
            for k, v in date_ranges.items()
        },
        'data_shapes': data_shapes,
        'scalers': scalers_info,
        'model_type': model_type or 'universal',
    }
    
    if data_counts:
        metadata['data_counts'] = data_counts
    
    return metadata
