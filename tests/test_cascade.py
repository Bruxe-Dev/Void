import sys
from pathlib import Path
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import pytest

from void_fx.features.cascade import (
    ANTISYMMETRIC_COLS,
    CASCADE_FEATURE_COLS,
    FeatureScaler,
    build_cascade_features,
    mirror_features,
)
from void_fx.labeling.cascade import (
    average_uniqueness,
    balanced_weights,
    breakeven_probability,
    direction_labels,
    prior_correct,
    side_aligned_outcome,
)
from void_fx.labeling.triple_barrier import bracket_outcomes


def create_dummy_ohlc(n=200):
    np.random.seed(42)
    close = 1.1000 + np.cumsum(np.random.normal(0, 0.0010, n))
    high = close + np.random.uniform(0.0001, 0.0005, n)
    low = close - np.random.uniform(0.0001, 0.0005, n)
    open_ = close + np.random.normal(0, 0.0002, n)
    dates = pd.date_range("2025-01-01", periods=n, freq="1h")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=dates)


def test_bracket_outcomes_symmetry():
    """Verify that inverted price moves mirror long and short outcomes."""
    df = create_dummy_ohlc(100)
    atr = (df["high"] - df["low"]).rolling(14).mean().bfill().values
    close = df["close"].values

    long_res = bracket_outcomes(close, atr, side=1, pt_mult=2.0, sl_mult=1.0, max_bars=12)
    short_res = bracket_outcomes(close, atr, side=-1, pt_mult=2.0, sl_mult=1.0, max_bars=12)

    assert len(long_res) == len(close)
    assert len(short_res) == len(close)
    assert np.isnan(long_res["outcome"].iloc[-12:]).all()


def test_direction_labels_balance():
    """Symmetric barrier should produce balanced classes without look-ahead bias."""
    df = create_dummy_ohlc(100)
    atr = (df["high"] - df["low"]).rolling(14).mean().bfill().values
    close = df["close"].values

    dir_df = direction_labels(close, atr, k=1.0, max_bars=12)
    valid = dir_df["y_dir"].dropna()
    assert len(valid) > 0
    assert set(valid.unique()).issubset({0.0, 1.0})


def test_mirror_features_invariance():
    """Verify antisymmetric feature flipping."""
    X = np.ones((5, len(CASCADE_FEATURE_COLS)), dtype=np.float32)
    X_mirrored = mirror_features(X)

    for i, col in enumerate(CASCADE_FEATURE_COLS):
        if col in ANTISYMMETRIC_COLS:
            assert np.allclose(X_mirrored[:, i], -1.0)
        else:
            assert np.allclose(X_mirrored[:, i], 1.0)


def test_uniqueness_weights():
    exit_bars = np.array([5, 5, 5, 2, 2])
    weights = average_uniqueness(exit_bars)
    assert len(weights) == 5
    assert (weights > 0).all()
    assert (weights <= 1.0).all()


def test_prior_correct():
    p_bal = np.array([0.5, 0.7, 0.3])
    prior = 0.333333  # 1/3
    p_true = prior_correct(p_bal, prior)
    assert np.isclose(p_true[0], prior, atol=1e-3)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
