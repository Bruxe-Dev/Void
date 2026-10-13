"""
Legacy (v1) feature pipeline: the exact 4 features the saved legacy models were
trained on. DO NOT change the math here; the pickled models depend on it.
"""

import numpy as np
import pandas as pd

from void_fx.config import ATR_PERIOD, LEGACY_FEATURE_COLS, LOOKBACK_WINDOW


def engineer_stationary_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Training-side transform from sprint1-data-engine.ipynb (cell 4).
    Returns close + the 4 legacy features, NaN warm-up rows dropped.
    """
    matrix = raw_df.copy()
    matrix["log_return"] = np.log(matrix["close"] / matrix["close"].shift(1))
    matrix["high_low_span"] = matrix["high"] - matrix["low"]
    matrix["volatility_atr"] = matrix["high_low_span"].rolling(window=ATR_PERIOD).mean()
    matrix["raw_hour"] = matrix.index.hour
    matrix["hour_sin"] = np.sin(2 * np.pi * matrix["raw_hour"] / 24.0)
    matrix["hour_cos"] = np.cos(2 * np.pi * matrix["raw_hour"] / 24.0)
    matrix.dropna(inplace=True)
    return matrix.drop(columns=["open", "high", "low", "raw_hour", "high_low_span"], errors="ignore")


def prepare_live_features(df: pd.DataFrame) -> tuple:
    """
    Transforms raw incoming OHLCV bars into aligned 2D (tabular) and 3D (temporal tensor) features.
    Maintains exact column sequence matching the training phase:
    ['log_return', 'volatility_atr', 'hour_cos', 'hour_sin']
    """
    matrix = df.copy()
    matrix['log_return'] = np.log(matrix['close'] / matrix['close'].shift(1))
    matrix['high_low_span'] = matrix['high'] - matrix['low']
    matrix['volatility_atr'] = matrix['high_low_span'].rolling(window=ATR_PERIOD).mean()
    matrix['hour'] = matrix.index.hour
    matrix['hour_sin'] = np.sin(2 * np.pi * matrix['hour'] / 24.0)
    matrix['hour_cos'] = np.cos(2 * np.pi * matrix['hour'] / 24.0)
    matrix.dropna(inplace=True)

    # IMPORTANT: Order must strictly match the training data split:
    X_2d = matrix[LEGACY_FEATURE_COLS]

    if len(X_2d) < LOOKBACK_WINDOW:
        raise ValueError(f"Insufficient historical bars: have {len(X_2d)}, required {LOOKBACK_WINDOW}")

    X_3d = np.array([X_2d.values[-LOOKBACK_WINDOW:]])

    return X_2d.tail(1), X_3d, matrix['volatility_atr'].iloc[-1]
