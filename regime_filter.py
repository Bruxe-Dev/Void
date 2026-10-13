"""
regime_filter.py
================
Step 1 of the upgrade plan: a causal, look-ahead-free market-regime filter.

Primary gate  : Wilder's Average Directional Index (ADX) with hysteresis.
Confirmation  : Rolling Hurst exponent (lagged-difference scaling method), optional.

Design rules
------------
* Every quantity at bar t uses ONLY bars <= t (ewm / rolling / shift(+k) only).
  There is no shift(-k), no centred window, no full-sample normalisation.
* The same functions are imported by the training notebook AND the live script,
  so train/serve skew is impossible by construction.
* Fail-safe: any NaN / warm-up / numerical problem resolves to "regime OFF"
  (i.e. Stand Down, Class 1). The filter can only REMOVE trades, never add them.
* Stateless: the hysteresis state is re-derived from the fetched bar history on
  every cycle, so a bot restart or a dropped MT5 connection cannot corrupt it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# DEFAULT PARAMETERS
# NOTE: thresholds MUST be calibrated on the TRAINING split only
#       (see regime_diagnostics below). These are textbook starting points.
# ------------------------------------------------------------------
ADX_PERIOD = 14
ADX_ENTER_THRESHOLD = 25.0   # regime switches ON when ADX rises to/above this
ADX_EXIT_THRESHOLD = 20.0    # regime switches OFF only when ADX falls below this
HURST_WINDOW = 128           # bars (~5.3 trading days on H1)
HURST_MAX_LAG = 20           # lags 2..20 used for the log-log scaling regression
HURST_THRESHOLD = 0.55       # > 0.5 = persistent; margin absorbs estimator noise
USE_HURST_CONFIRMATION = True
# Synthetic sweep (20k H1 bars each), share of bars with the gate ON:
#   ADX 25/20 alone      -> random walk 49% | OU chop 28% | trend 93%
#   ADX 25/20 + H>0.55   -> random walk 18% | OU chop  2% | trend 90%
# ADX alone cannot separate a noisy random walk from a weak trend; Hurst can.

# Wilder smoothing has infinite memory: residual weight of the seed after k bars
# is (1 - 1/n)^k. For n=14, k=140 -> 3.2e-5, k=400 -> 1.4e-13. We require at
# least 10*n bars after warm-up so live values match training values.
MIN_CONVERGENCE_BARS = 10 * ADX_PERIOD


# ==================================================================
# 1. WILDER ADX
# ==================================================================
def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's RMA: y_t = y_{t-1} + (x_t - y_{t-1}) / n  ==  EMA with alpha = 1/n.
    Strictly causal (adjust=False -> pure recursion, no future weights).
    """
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """
    Computes +DI, -DI, DX and ADX from OHLC bars.

    Parameters
    ----------
    df : DataFrame with columns ['high', 'low', 'close'], chronologically sorted.

    Returns
    -------
    DataFrame (same index) with columns ['plus_di', 'minus_di', 'dx', 'adx', 'true_range'].
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    # --- Directional movement -------------------------------------
    up_move = high.diff()          # H_t - H_{t-1}
    down_move = -low.diff()        # L_{t-1} - L_t

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )
    # First bar has no predecessor -> undefined, not zero.
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan

    # --- True Range (includes overnight / weekend gaps) -------------
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1, skipna=False)

    # --- Wilder smoothing -----------------------------------------
    atr_w = _wilder_smooth(true_range, period)
    plus_dm_s = _wilder_smooth(plus_dm, period)
    minus_dm_s = _wilder_smooth(minus_dm, period)

    # Guard against zero ATR (frozen feed / holiday bars) -> NaN, never inf.
    atr_safe = atr_w.replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm_s / atr_safe
    minus_di = 100.0 * minus_dm_s / atr_safe

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum

    adx = _wilder_smooth(dx, period)

    return pd.DataFrame(
        {
            "plus_di": plus_di,
            "minus_di": minus_di,
            "dx": dx,
            "adx": adx,
            "true_range": true_range,
        },
        index=df.index,
    )


# ==================================================================
# 2. ROLLING HURST EXPONENT (lagged-difference / variogram scaling)
# ==================================================================
def compute_rolling_hurst(
    close: pd.Series, window: int = HURST_WINDOW, max_lag: int = HURST_MAX_LAG
) -> pd.Series:
    """
    For fractional Brownian motion, Std[x_{t+tau} - x_t] ∝ tau^H.
    At every bar t we estimate H as the OLS slope of
        log Std_tau(t)  on  log tau,   tau = 2..max_lag,
    where Std_tau(t) is the rolling std of tau-lag log-price differences over
    the trailing `window` bars. Fully vectorised and causal.

    H ≈ 0.5 random walk | H > 0.5 persistent / trending | H < 0.5 mean-reverting.
    """
    if max_lag >= window // 2:
        raise ValueError("max_lag must be well below window/2 for a stable estimate.")

    log_p = np.log(close.astype(float))
    lags = np.arange(2, max_lag + 1)
    log_lags = np.log(lags)
    x_c = log_lags - log_lags.mean()
    denom = np.sum(x_c ** 2)

    # Column k = log RMS of tau_k differences over the trailing window (causal).
    # We deliberately use the UNCENTRED second moment (variogram), not .std():
    #   E[d_tau^2] = sigma^2 * tau            (random walk  -> slope 0.5, unbiased)
    #   E[d_tau^2] = sigma^2 * tau + mu^2 tau^2 (drift      -> slope > 0.5)
    # .std() subtracts the window mean of d_tau, which (a) deletes the drift
    # term mu*tau and (b) biases H downward because overlapping differences
    # share the same endpoints. Both are fatal for a trend detector.
    log_std = pd.concat(
        {
            int(tau): 0.5 * np.log(
                ((log_p - log_p.shift(int(tau))) ** 2)
                .rolling(window - int(tau), min_periods=window - int(tau))
                .mean()
                .replace(0.0, np.nan)
            )
            for tau in lags
        },
        axis=1,
    )

    # Closed-form OLS slope across the lag dimension, row by row.
    y_c = log_std.sub(log_std.mean(axis=1), axis=0)
    hurst = (y_c * x_c).sum(axis=1, min_count=len(lags)) / denom
    return hurst.rename("hurst")


# ==================================================================
# 3. REGIME STATE MACHINE (hysteresis)
# ==================================================================
def _hysteresis_state(adx: np.ndarray, enter: float, exit_: float) -> np.ndarray:
    """
    Schmitt-trigger: ON when adx >= enter, stays ON until adx < exit_.
    NaN always forces OFF (fail-safe). O(N) loop; ~50 ms for 100k bars.
    """
    if exit_ > enter:
        raise ValueError("exit threshold must be <= enter threshold.")
    state = np.zeros(adx.shape[0], dtype=bool)
    on = False
    for i, v in enumerate(adx):
        if np.isnan(v):
            on = False
        elif not on and v >= enter:
            on = True
        elif on and v < exit_:
            on = False
        state[i] = on
    return state


def compute_regime_features(
    df: pd.DataFrame,
    adx_period: int = ADX_PERIOD,
    adx_enter: float = ADX_ENTER_THRESHOLD,
    adx_exit: float = ADX_EXIT_THRESHOLD,
    use_hurst_confirmation: bool = USE_HURST_CONFIRMATION,
    hurst_window: int = HURST_WINDOW,
    hurst_max_lag: int = HURST_MAX_LAG,
    hurst_threshold: float = HURST_THRESHOLD,
) -> pd.DataFrame:
    """
    Single entry point used by BOTH training and live code.

    Returns a DataFrame aligned to df.index with:
        adx, plus_di, minus_di, hurst, adx_regime_on, regime_on (final gate, bool)
    """
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"regime filter requires columns {missing}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Bars must be sorted chronologically (oldest first).")

    out = compute_adx(df, adx_period)[["adx", "plus_di", "minus_di"]].copy()
    out["hurst"] = compute_rolling_hurst(df["close"], hurst_window, hurst_max_lag)

    out["adx_regime_on"] = _hysteresis_state(out["adx"].to_numpy(), adx_enter, adx_exit)

    # Mask the convergence period: values here differ between a 500-bar live
    # fetch and a multi-year training history, so treat them as unknown -> OFF.
    warmup = 2 * adx_period + MIN_CONVERGENCE_BARS
    out.iloc[:warmup, out.columns.get_loc("adx_regime_on")] = False

    gate = out["adx_regime_on"].copy()
    if use_hurst_confirmation:
        gate &= out["hurst"].gt(hurst_threshold).fillna(False)
    out["regime_on"] = gate.astype(bool)
    return out


# ==================================================================
# 4. LIVE HELPER
# ==================================================================
@dataclass
class RegimeDecision:
    regime_on: bool
    adx: float = float("nan")
    plus_di: float = float("nan")
    minus_di: float = float("nan")
    hurst: float = float("nan")
    bar_time: object = None
    reason: str = ""
    extra: dict = field(default_factory=dict)


def evaluate_live_regime(df_closed_bars: pd.DataFrame, **kwargs) -> RegimeDecision:
    """
    Evaluates the gate on the most recent CLOSED bar.

    IMPORTANT: pass only closed bars. mt5.copy_rates_from_pos(..., 0, N) returns
    the currently-forming candle at the last row; drop it (df.iloc[:-1]) or the
    live ADX will be computed on a partial bar that never existed in training.

    Never raises: any failure returns regime_on=False (Stand Down).
    """
    try:
        min_bars = 2 * ADX_PERIOD + MIN_CONVERGENCE_BARS + 1
        if len(df_closed_bars) < min_bars:
            return RegimeDecision(False, reason=f"insufficient bars ({len(df_closed_bars)} < {min_bars})")

        feats = compute_regime_features(df_closed_bars, **kwargs)
        last = feats.iloc[-1]
        on = bool(last["regime_on"])
        if not np.isfinite(last["adx"]):
            on, reason = False, "ADX is NaN"
        else:
            reason = "TRENDING" if on else "RANGING / CHOP"
        return RegimeDecision(
            regime_on=on,
            adx=float(last["adx"]),
            plus_di=float(last["plus_di"]),
            minus_di=float(last["minus_di"]),
            hurst=float(last["hurst"]),
            bar_time=feats.index[-1],
            reason=reason,
        )
    except Exception as exc:  # fail-safe: never let the filter crash the loop
        return RegimeDecision(False, reason=f"regime filter error: {exc!r}")


# ==================================================================
# 5. CALIBRATION DIAGNOSTICS (training split ONLY)
# ==================================================================
def regime_diagnostics(regime: pd.DataFrame, labels: pd.Series, bins=(0, 15, 20, 25, 30, 40, 100)) -> pd.DataFrame:
    """
    Cross-tabulates your existing Triple-Barrier labels (0=SL, 1=Expiry, 2=TP)
    against ADX buckets. Run this on the TRAINING split only to pick thresholds;
    picking them on validation/test data is a form of leakage.

    Under a driftless random walk with a 2:1 bracket, P(TP) -> 1/3. Buckets whose
    TP share is ~0.33 (minus costs) carry no edge and should be gated out.
    """
    aligned = pd.concat([regime["adx"], labels.rename("label")], axis=1, join="inner").dropna()
    aligned["adx_bucket"] = pd.cut(aligned["adx"], bins=bins)
    table = pd.crosstab(aligned["adx_bucket"], aligned["label"], normalize="index").round(3)
    table["n_samples"] = aligned.groupby("adx_bucket", observed=False).size()
    return table
