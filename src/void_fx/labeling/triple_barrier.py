"""
Triple-barrier labelling.

* generate_triple_barrier_labels  -> ORIGINAL long-only labeller from the notebook,
                                     kept verbatim so legacy results stay reproducible.
* bracket_outcomes                -> vectorised, side-aware (long OR short) bracket
                                     simulator used by the Step 2 cascade.

All barriers are evaluated on CLOSES of bars t+1 .. t+max_bars (same convention as
the notebook), so the label at bar t uses only information strictly after t and
the feature row at t uses only information up to and including t.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from void_fx.config import MAX_HOLDING_BARS, PT_MULT, SL_MULT

OUTCOME_TP, OUTCOME_EXPIRY, OUTCOME_SL = 1, 0, -1


def generate_triple_barrier_labels(df: pd.DataFrame, pt_mult: float = 2.0, sl_mult: float = 1.0, max_bars: int = 24) -> pd.Series:
    """LEGACY (notebook cell 6): long-only labels, +1 TP / -1 SL / 0 expiry."""
    labels = np.zeros(len(df))

    close_prices = df['close'].values
    atr_values = df['volatility_atr'].values

    total_rows = len(df)

    for t in range(total_rows - max_bars):
        current_close = close_prices[t]
        current_atr = atr_values[t]

        upper_barrier = current_close + (current_atr * pt_mult)
        lower_barrier = current_close - (current_atr * sl_mult)

        forward_path = close_prices[t + 1: t + 1 + max_bars]

        profit_crossings = np.where(forward_path >= upper_barrier)[0]
        loss_crossings = np.where(forward_path <= lower_barrier)[0]

        first_profit_idx = profit_crossings[0] if len(profit_crossings) > 0 else None
        first_loss_idx = loss_crossings[0] if len(loss_crossings) > 0 else None

        if first_profit_idx is not None and first_loss_idx is not None:
            if first_profit_idx < first_loss_idx:
                labels[t] = 1
            else:
                labels[t] = -1
        elif first_profit_idx is not None:
            labels[t] = 1
        elif first_loss_idx is not None:
            labels[t] = -1
        else:
            labels[t] = 0

    return pd.Series(labels, index=df.index, name='target')


def bracket_outcomes(
    close: np.ndarray,
    atr: np.ndarray,
    side: int,
    pt_mult: float = PT_MULT,
    sl_mult: float = SL_MULT,
    max_bars: int = MAX_HOLDING_BARS,
) -> pd.DataFrame:
    """
    Simulates a bracket opened at close[t] for every t, in the given direction.

    side = +1 (long):  TP at close + pt*ATR, SL at close - sl*ATR
    side = -1 (short): TP at close - pt*ATR, SL at close + sl*ATR

    Returns a DataFrame (positional, len(close) rows) with:
        r        : realised R-multiple (+pt/sl, -1, or mark-to-market at expiry), before costs
        outcome  : +1 TP, -1 SL, 0 expiry
        exit_bar : bars until the position closes (1..max_bars); used for overlap weights
    The last `max_bars` rows are NaN because their future is unknown.
    """
    if side not in (1, -1):
        raise ValueError("side must be +1 (long) or -1 (short)")
    close = np.asarray(close, dtype=float)
    atr = np.asarray(atr, dtype=float)
    n = len(close)
    r = np.full(n, np.nan)
    outcome = np.full(n, np.nan)
    exit_bar = np.full(n, np.nan)
    if n <= max_bars:
        return pd.DataFrame({"r": r, "outcome": outcome, "exit_bar": exit_bar})

    fwd = sliding_window_view(close[1:], max_bars)        # row t -> close[t+1 : t+1+max_bars]
    m = fwd.shape[0]                                       # = n - max_bars
    c, a = close[:m, None], atr[:m, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        move = side * (fwd - c) / a                        # favourable excursion in ATR units

    hit_tp = move >= pt_mult
    hit_sl = move <= -sl_mult
    never = max_bars + 1
    first_tp = np.where(hit_tp.any(1), hit_tp.argmax(1), never)
    first_sl = np.where(hit_sl.any(1), hit_sl.argmax(1), never)

    tp_first = first_tp < first_sl
    sl_first = first_sl < first_tp
    out = np.where(tp_first, OUTCOME_TP, np.where(sl_first, OUTCOME_SL, OUTCOME_EXPIRY))
    rr = np.where(tp_first, pt_mult / sl_mult, np.where(sl_first, -1.0, move[:, -1] / sl_mult))
    eb = np.minimum(np.minimum(first_tp, first_sl), max_bars - 1) + 1

    valid = np.isfinite(atr[:m]) & (atr[:m] > 0)
    r[:m] = np.where(valid, rr, np.nan)
    outcome[:m] = np.where(valid, out, np.nan)
    exit_bar[:m] = np.where(valid, eb, np.nan)
    return pd.DataFrame({"r": r, "outcome": outcome, "exit_bar": exit_bar})
