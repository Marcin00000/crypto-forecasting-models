"""
Core configuration for data preparation module
"""
import os


class PrepConfig:
    """Configuration for data preparation"""
    
    # Default parameters
    DEFAULT_LOOKBACK = 48
    DEFAULT_HORIZON = 1
    DEFAULT_FEATURE_SET = 'full'
    
    # Split modes
    SPLIT_MODE_MONTHS = 'months'
    SPLIT_MODE_PERCENTAGE = 'percentage'
    SPLIT_MODE_DATES = 'dates'
    
    # Data types to prepare
    DATA_TYPE_TRAIN = 'train'
    DATA_TYPE_AGENT = 'agent'
    DATA_TYPE_TEST = 'test'
    DATA_TYPE_SYM = 'sym'
    
    # Output formats
    FORMAT_NATIVE = 'native'
    FORMAT_CSV = 'csv'
    FORMAT_BOTH = 'both'
    
    # Scalers
    SCALER_MINMAX = 'minmax'
    SCALER_STANDARD = 'standard'
    SCALER_ROBUST = 'robust'
    SCALER_NONE = 'none'
    SCALER_QUANTILE_NORMAL = 'quantile_normal'
    
    # Symbol ID mapping
    SYMBOL_ID_MAP = {
        'btc': 'B', 'eth': 'E', 'bnb': 'N', 'sol': 'S', 'ada': 'A',
        'xrp': 'X', 'doge': 'D', 'dot': 'P', 'matic': 'M', 'avax': 'V'
    }
    
    # Interval mapping
    INTERVAL_MAP = {
        60: '1m', 300: '5m', 900: '15m', 1800: '30m',
        3600: '1h', 14400: '4h', 86400: '1d'
    }
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = 'prepared_data'
    
    # Model adapters registry
    SUPPORTED_MODELS = ['lstm', 'xgb']
    
    @staticmethod
    def get_symbol_prefix(symbol):
        """Get data ID prefix based on symbol (not model)"""
        return PrepConfig.SYMBOL_ID_MAP.get(symbol.lower(), 'C')
