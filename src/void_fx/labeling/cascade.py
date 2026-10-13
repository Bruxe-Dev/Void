"""
Step 2: Two-stage binomial cascade labels (meta-labelling).

Stage 1 - DIRECTION (primary model): "Which way does price break first?"
    A SYMMETRIC bracket of +/-k*ATR. Up-first -> 1, down-first -> 0, neither -> dropped.
    Symmetric barriers mean that, under no drift, P(up) = P(down) = 0.5, so the
    dataset is balanced BY CONSTRUCTION (not by resampling). Mirror augmentation
    (see features.cascade.mirror_features) then makes the balance exact.

Stage 2 - RISK (meta model): "Given the side the primary model chose, does the
    real 2:1 bracket in THAT direction hit its take-profit?"  1 = TP, 0 = SL/expiry.
    This is where the 2:1 asymmetry lives. Its natural base rate is ~1/3, so it is
    trained with class-balanced weights and its probabilities are then mapped back
    to true probabilities with an exact prior-shift correction (prior_correct).

Separating "direction" from "risk" is what lets the system short: the primary
model is symmetric, and the meta model simply decides whether the proposed trade,
long or short, is worth taking.
"""

import numpy as np
import pandas as pd

from void_fx.config import MAX_HOLDING_BARS
from void_fx.labeling.triple_barrier import OUTCOME_SL, OUTCOME_TP, bracket_outcomes

DIRECTION_BARRIER_MULT = 1.0   # symmetric +/- 1 ATR for the direction label


def direction_labels(close, atr, k: float = DIRECTION_BARRIER_MULT, max_bars: int = MAX_HOLDING_BARS) -> pd.DataFrame:
    """
    Returns positional DataFrame with:
        y_dir    : 1.0 (up first), 0.0 (down first), NaN (no touch or unknown future)
        exit_bar : bars until the symmetric barrier was touched (for uniqueness weights)
    """
    b = bracket_outcomes(close, atr, side=1, pt_mult=k, sl_mult=k, max_bars=max_bars)
    y = np.where(b["outcome"] == OUTCOME_TP, 1.0, np.where(b["outcome"] == OUTCOME_SL, 0.0, np.nan))
    return pd.DataFrame({"y_dir": y, "exit_bar": b["exit_bar"].to_numpy()})


def side_aligned_outcome(side: np.ndarray, long_df: pd.DataFrame, short_df: pd.DataFrame) -> pd.DataFrame:
    """
    Picks the long or short bracket result according to the proposed side (+1/-1).
    Returns columns r (R-multiple before costs), y_meta (1 = TP hit), exit_bar.
    """
    side = np.asarray(side)
    is_long = side > 0
    r = np.where(is_long, long_df["r"], short_df["r"])
    outcome = np.where(is_long, long_df["outcome"], short_df["outcome"])
    exit_bar = np.where(is_long, long_df["exit_bar"], short_df["exit_bar"])
    y = np.where(np.isnan(outcome), np.nan, (outcome == OUTCOME_TP).astype(float))
    return pd.DataFrame({"r": r, "y_meta": y, "exit_bar": exit_bar})


def average_uniqueness(exit_bar: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Lopez de Prado's average uniqueness. Label i is "alive" on bars i+1 .. i+exit_bar[i].
    Overlapping labels share information; each label's weight is the mean of
    1/concurrency over its lifespan. Without this, 24 consecutive bars that all
    resolve on the same price move count as 24 independent examples, which is the
    main reason the legacy models memorised their training set.

    mask: optional boolean array; only masked labels count towards concurrency
          (used to measure overlap among actually-taken trades).
    Returns weights in (0, 1]; NaN where exit_bar is NaN or masked out.
    """
    e = np.asarray(exit_bar, dtype=float)
    n = len(e)
    active = np.isfinite(e) if mask is None else (np.isfinite(e) & np.asarray(mask, bool))
    idx = np.flatnonzero(active)
    ends = idx + e[idx].astype(int)                       # inclusive end bar

    diff = np.zeros(n + 2)
    np.add.at(diff, idx + 1, 1)
    np.add.at(diff, np.minimum(ends + 1, n + 1), -1)
    conc = np.cumsum(diff)[: n + 1]                       # concurrency at bar t (0..n)
    inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1), 0.0)
    csum = np.concatenate([[0.0], np.cumsum(inv)])        # csum[t] = sum inv[0..t-1]

    w = np.full(n, np.nan)
    ends_c = np.minimum(ends, n)
    w[idx] = (csum[ends_c + 1] - csum[idx + 1]) / (ends_c - idx)
    return w


def balanced_weights(y: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
    """Class-balanced weights (each class gets equal total weight), multiplied by `base`."""
    y = np.asarray(y)
    base = np.ones(len(y)) if base is None else np.asarray(base, float)
    w = np.zeros(len(y))
    for cls in np.unique(y):
        m = y == cls
        w[m] = base[m] / base[m].sum()
    return w * len(y) / w.sum()


def prior_correct(p_balanced: np.ndarray, prior: float) -> np.ndarray:
    """
    Exact prior-shift correction. A model trained on class-balanced weights
    estimates odds_bal = odds_true * (0.5/0.5) / (pi/(1-pi)). Inverting:
        odds_true = odds_bal * pi / (1 - pi)
    so we get balanced training (no "lazy brain") AND honest probabilities for
    the expected-value threshold.
    """
    p = np.clip(np.asarray(p_balanced, float), 1e-6, 1 - 1e-6)
    odds = p / (1 - p) * prior / (1 - prior)
    return odds / (1 + odds)


def breakeven_probability(cost_r, pt_over_sl: float = 2.0) -> np.ndarray:
    """
    p* such that  p*R_tp - (1-p)*1 - cost = 0  ->  p* = (1 + cost) / (1 + R_tp).
    For a 2:1 bracket with zero cost this is the familiar 1/3.
    """
    return (1.0 + np.asarray(cost_r, float)) / (1.0 + pt_over_sl)
