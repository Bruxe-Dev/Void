"""
Step 2 feature set (v2) for the bidirectional cascade.

Every feature is either
    ANTISYMMETRIC : flips sign if the price chart is turned upside down (returns,
                    +DI minus -DI, position in range), or
    SYMMETRIC     : unchanged by that flip (volatility, ADX, Hurst, time of day).

That split lets us (a) mirror-augment the training data, so the direction model
sees exactly as many down-moves as up-moves, and (b) symmetrise inference so the
model cannot develop a permanent long or short bias. Turning the chart upside down
is economically valid in FX: an inverted EURUSD chart *is* the USDEUR market.

All features are causal (rolling/ewm/shift(+k) only). Returns are scaled by ATR%,
which makes EURUSD and GBPUSD comparable, so one model can be trained on both.
"""

import numpy as np
import pandas as pd

from void_fx.config import ATR_PERIOD
from void_fx.features.regime import compute_regime_features

ANTISYMMETRIC_COLS = ["ret_1", "ret_4", "ret_24", "di_spread", "range_pos_24"]
SYMMETRIC_COLS = ["adx_n", "hurst", "log_atr_pct", "atr_ratio", "hour_sin", "hour_cos"]
CASCADE_FEATURE_COLS = ANTISYMMETRIC_COLS + SYMMETRIC_COLS

# Longest look-back: regime filter (169 bars) + atr_ratio window (120) -> ~300 bars.
MIN_HISTORY_BARS = 320


def build_cascade_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    raw: OHLC(+spread) bars for ONE symbol, chronological, CLOSED bars only.
    Returns raw columns + 'volatility_atr' (legacy ATR, used for barrier sizing,
    identical to what the live script uses for SL/TP) + CASCADE_FEATURE_COLS.
    Rows are NOT dropped; warm-up rows contain NaN.
    """
    if not raw.index.is_monotonic_increasing:
        raise ValueError("Bars must be chronological.")
    df = raw.copy()
    close = df["close"].astype(float)

    df["volatility_atr"] = (df["high"] - df["low"]).rolling(ATR_PERIOD).mean()
    atr_pct = (df["volatility_atr"] / close).replace(0.0, np.nan)

    log_ret = np.log(close / close.shift(1))
    df["ret_1"] = log_ret / atr_pct
    df["ret_4"] = log_ret.rolling(4).sum() / atr_pct
    df["ret_24"] = log_ret.rolling(24).sum() / atr_pct

    hi24, lo24 = df["high"].rolling(24).max(), df["low"].rolling(24).min()
    span = (hi24 - lo24).replace(0.0, np.nan)
    df["range_pos_24"] = 2.0 * (close - lo24) / span - 1.0          # -1 at 24h low, +1 at 24h high

    reg = compute_regime_features(df[["high", "low", "close"]], use_hurst_confirmation=False)
    df["di_spread"] = (reg["plus_di"] - reg["minus_di"]) / 100.0
    df["adx_n"] = reg["adx"] / 100.0
    df["hurst"] = reg["hurst"]

    df["log_atr_pct"] = np.log(atr_pct)
    df["atr_ratio"] = df["volatility_atr"] / df["volatility_atr"].rolling(120).mean()

    hour = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return df


def mirror_features(X: np.ndarray, feature_cols: list[str] = CASCADE_FEATURE_COLS) -> np.ndarray:
    """
    Upside-down chart: negates antisymmetric columns along the LAST axis.
    Works for 2D [rows, features] and 3D [samples, time, features] arrays.
    Requires antisymmetric columns to be zero-centred (see FeatureScaler).
    """
    Xm = np.array(X, dtype=np.float32, copy=True)
    anti_idx = [feature_cols.index(c) for c in ANTISYMMETRIC_COLS if c in feature_cols]
    Xm[..., anti_idx] *= -1.0
    return Xm


class FeatureScaler:
    """
    Z-score scaler fitted on the TRAINING segment only. Antisymmetric columns are
    scaled but NOT centred (mean forced to 0) so mirroring stays exact after scaling.
    """

    def __init__(self, feature_cols: list[str] = CASCADE_FEATURE_COLS):
        self.feature_cols = list(feature_cols)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        X = np.asarray(X, float)
        mean = np.nanmean(X, axis=0)
        anti = [i for i, c in enumerate(self.feature_cols) if c in ANTISYMMETRIC_COLS]
        mean[anti] = 0.0
        std = np.sqrt(np.nanmean((X - mean) ** 2, axis=0))
        std[~np.isfinite(std) | (std < 1e-12)] = 1.0
        self.mean_, self.std_ = mean, std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, float) - self.mean_) / self.std_).astype(np.float32)

    def to_dict(self) -> dict:
        return {"feature_cols": self.feature_cols, "mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        s = cls(d["feature_cols"])
        s.mean_, s.std_ = np.asarray(d["mean"]), np.asarray(d["std"])
        return s
