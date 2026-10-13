"""
calibrate_regime.py
===================
Calibrates the regime-filter thresholds (ADX enter/exit, Hurst) on the TRAINING
split only, replicating sprint1-data-engine.ipynb exactly:
    * 30,000 H1 bars per symbol
    * Triple barrier on closes: TP = 2*ATR, SL = 1*ATR, 24-bar expiry
    * ATR = rolling-14 mean of (high - low)  (same as the notebook)
    * Chronological 80/20 split with a 24-bar purge gap

Read-only: it only downloads historical bars, it never sends orders.

Run with the project environment:
    C:\\Users\\HP\\anaconda3\\envs\\void\\python.exe calibrate_regime.py
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from regime_filter import compute_regime_features

SYMBOLS = ["EURUSD", "GBPUSD"]
NUM_BARS = 30_000
MAX_BARS = 24
PT_MULT, SL_MULT = 2.0, 1.0
TRAIN_RATIO = 0.8
MIN_COVERAGE = 0.10          # gate must allow at least 10% of bars, else too few trades
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
CACHE_DIR = Path("data-engine")


# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
def fetch_bars(symbol: str) -> tuple[pd.DataFrame, float]:
    """Fetches CLOSED bars (start_pos=1 skips the forming candle). Cached to parquet/csv."""
    cache = CACHE_DIR / f"{symbol}_H1_{NUM_BARS}.csv"
    meta = CACHE_DIR / f"{symbol}_meta.json"
    if cache.exists() and meta.exists():
        df = pd.read_csv(cache, index_col="time", parse_dates=True)
        return df, json.loads(meta.read_text())["point"]

    import MetaTrader5 as mt5
    if not mt5.initialize(path=MT5_PATH):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 1, NUM_BARS)
        info = mt5.symbol_info(symbol)
        if rates is None or info is None:
            raise RuntimeError(f"No data for {symbol}: {mt5.last_error()}")
        point = info.point
    finally:
        mt5.shutdown()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")[["open", "high", "low", "close", "tick_volume", "spread"]]
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_csv(cache)
    meta.write_text(json.dumps({"point": point}))
    return df, point


# ------------------------------------------------------------------
# LABELS (both directions, R-multiple outcomes)
# ------------------------------------------------------------------
def barrier_outcomes(close: np.ndarray, atr: np.ndarray, long: bool) -> np.ndarray:
    """
    R-multiple of a bracket opened at close[t]: +2 (TP), -1 (SL) or the
    marked-to-market move at expiry (in ATR units). Identical crossing logic
    to generate_triple_barrier_labels() in the notebook, mirrored for shorts.
    Last MAX_BARS rows are NaN (future unknown).
    """
    n = len(close)
    out = np.full(n, np.nan)
    fwd = sliding_window_view(close[1:], MAX_BARS)          # row t -> close[t+1 : t+1+MAX_BARS]
    m = fwd.shape[0] - 0
    c, a = close[:m], atr[:m]
    sign = 1.0 if long else -1.0
    move = sign * (fwd - c[:, None]) / a[:, None]          # favourable move in ATR units
    hit_tp = move >= PT_MULT
    hit_sl = move <= -SL_MULT
    first_tp = np.where(hit_tp.any(1), hit_tp.argmax(1), MAX_BARS + 1)
    first_sl = np.where(hit_sl.any(1), hit_sl.argmax(1), MAX_BARS + 1)
    r = np.where(first_tp < first_sl, PT_MULT,
                 np.where(first_sl < first_tp, -SL_MULT, move[:, -1]))   # tie impossible except both none
    out[:m] = r
    out[n - MAX_BARS:] = np.nan
    return out


def build_dataset(symbol: str) -> pd.DataFrame:
    raw, point = fetch_bars(symbol)
    df = raw.copy()
    df["volatility_atr"] = (df["high"] - df["low"]).rolling(14).mean()
    regime = compute_regime_features(df[["high", "low", "close"]], use_hurst_confirmation=False)
    df = df.join(regime[["adx", "plus_di", "minus_di", "hurst"]])
    df = df.dropna(subset=["volatility_atr"])

    c, a = df["close"].to_numpy(), df["volatility_atr"].to_numpy()
    df["r_long"] = barrier_outcomes(c, a, long=True)
    df["r_short"] = barrier_outcomes(c, a, long=False)
    df["cost_r"] = df["spread"] * point / df["volatility_atr"]   # spread paid, in R
    df = df.iloc[:-MAX_BARS]                                     # same as notebook's iloc[:-24]
    df["symbol"] = symbol
    return df


def purged_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = int(len(df) * TRAIN_RATIO)
    return df.iloc[:split], df.iloc[split + MAX_BARS:]


# ------------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------------
def evaluate(df: pd.DataFrame, enter, exit_, hurst_thr) -> dict:
    """
    Evaluates a gate on pre-computed ADX/Hurst. The hysteresis is recomputed per
    symbol over the full causal history, then masked to the requested rows.
    Score = 'follow-the-DI' naive strategy: long if +DI > -DI else short, only
    when the gate is ON, net of spread. It is model-free, so it measures the
    regime filter itself, not the ML models.
    """
    on = df["gate"].to_numpy()
    sub = df[on]
    n, cov = len(sub), on.mean()
    if n < 200:
        return {"n": n, "coverage": cov}
    go_long = sub["plus_di"] > sub["minus_di"]
    r = np.where(go_long, sub["r_long"], sub["r_short"]) - sub["cost_r"]
    tp_any = ((sub["r_long"] == PT_MULT) | (sub["r_short"] == PT_MULT)).mean()
    sl_long = (sub["r_long"] == -SL_MULT).mean()
    # Overlapping 24h labels -> effective sample size ~ n / 24 for the t-stat.
    t_stat = r.mean() / (r.std(ddof=1) / np.sqrt(max(n / MAX_BARS, 1)))
    return {"n": n, "coverage": cov, "ev_r": r.mean(), "t_eff": t_stat,
            "p_tp_either": tp_any, "p_sl_long": sl_long,
            "p_tp_followed": (r + sub["cost_r"] >= PT_MULT - 1e-9).mean()}


def apply_gate(full: pd.DataFrame, enter, exit_, hurst_thr) -> pd.Series:
    from regime_filter import _hysteresis_state
    parts = []
    for _, g in full.groupby("symbol", sort=False):
        st = _hysteresis_state(g["adx"].to_numpy(), enter, exit_)
        if hurst_thr is not None:
            st &= g["hurst"].gt(hurst_thr).to_numpy()
        parts.append(pd.Series(st, index=g.index))
    return pd.concat(parts)


def main():
    frames = {s: build_dataset(s) for s in SYMBOLS}
    for s, d in frames.items():
        tr, te = purged_split(d)
        print(f"{s}: {len(d)} rows | train {tr.index[0]} -> {tr.index[-1]} | test {te.index[0]} -> {te.index[-1]}")

    grid = list(itertools.product([20, 22, 25, 28, 30, 35], [None, 0.50, 0.52, 0.55, 0.58, 0.60]))
    rows = []
    for enter, h in grid:
        exit_ = enter - 5
        tr_all = []
        for s, d in frames.items():
            d = d.copy()
            d["gate"] = apply_gate(d, enter, exit_, h).to_numpy()
            tr, _ = purged_split(d)
            tr_all.append(tr)
        res = evaluate(pd.concat(tr_all), enter, exit_, h)
        rows.append({"enter": enter, "exit": exit_, "hurst": h, **res})

    # Baseline: no gate at all (every bar)
    base = []
    for s, d in frames.items():
        d = d.copy(); d["gate"] = True
        base.append(purged_split(d)[0])
    b = evaluate(pd.concat(base), None, None, None)

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== BASELINE (no filter), TRAIN split ===")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in b.items()})
    print("\n=== GRID, TRAIN split only (sorted by effective t-stat) ===")
    eligible = table[table["coverage"] >= MIN_COVERAGE].sort_values("t_eff", ascending=False)
    print(eligible.round(4).to_string(index=False))

    best = eligible.iloc[0]
    print(f"\n>>> SELECTED on TRAIN: enter={best.enter}, exit={best.exit}, hurst={best.hurst}")

    # One-shot out-of-sample confirmation (NOT used for selection).
    te_all, te_base = [], []
    for s, d in frames.items():
        d = d.copy()
        d["gate"] = apply_gate(d, best.enter, best.exit,
                               None if pd.isna(best.hurst) else best.hurst).to_numpy()
        te_all.append(purged_split(d)[1])
        d2 = d.copy(); d2["gate"] = True
        te_base.append(purged_split(d2)[1])
    print("\n=== OUT-OF-SAMPLE TEST (looked at once, after selection) ===")
    for name, part in [("no filter", te_base), ("selected filter", te_all)]:
        r = evaluate(pd.concat(part), None, None, None)
        print(f"{name:16s}", {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})

    Path("regime_calibration.json").write_text(json.dumps({
        "adx_enter": float(best.enter), "adx_exit": float(best.exit),
        "hurst_threshold": None if pd.isna(best.hurst) else float(best.hurst),
        "train_metrics": {k: float(best[k]) for k in ["n", "coverage", "ev_r", "t_eff", "p_tp_either"]},
    }, indent=2))
    print("\nSaved regime_calibration.json")


if __name__ == "__main__":
    sys.exit(main())
