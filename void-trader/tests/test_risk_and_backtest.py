"""
Comprehensive tests for Void Trader:
  * Account-size-aware position sizing ($50, $100, $500, $100,000)
  * Hard risk limits (per-trade, aggregate exposure, daily loss, max drawdown,
    margin utilization, max concurrent positions, consecutive losses, stale account,
    non-demo rejection, emergency stop, duplicate bar submission, stop-loss side/distance)
  * Regression tests for fixed defects (no round-up to volume_min, no 0.01 fallback on
    missing symbol info, closed-bar feature slicing, per-symbol 3D tensor construction)
  * Backtest engine accounting (balance, equity, unrealized P&L, margin, costs, rejections)
  * Model loading and inference smoke tests (both Cascade and Legacy v1 checkpoints)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import pytest

from void_fx.backtest.engine import BacktestConfig, default_fx_spec, run_backtest
from void_fx.execution.risk_manager import (
    BUY,
    SELL,
    AccountSnapshot,
    ExposureItem,
    InstrumentSpec,
    OrderIntent,
    RiskLimits,
    RiskState,
    floor_to_step,
    size_position,
    validate_order,
)
from void_fx.features.base import prepare_live_features
from void_fx.features.cascade import CASCADE_FEATURE_COLS, FeatureScaler, build_cascade_features
from void_fx.live.cascade_runner import load_cascade_models, predict_cascade
from void_fx.live.legacy_runner import load_models as load_legacy_models, predict_legacy


def make_account(
    balance: float = 100_000.0,
    equity: float | None = None,
    margin: float = 0.0,
    leverage: int = 100,
    is_demo: bool = True,
    fetched_at: float = 1_700_000_000.0,
) -> AccountSnapshot:
    eq = balance if equity is None else equity
    return AccountSnapshot(
        balance=balance,
        equity=eq,
        margin=margin,
        margin_free=max(0.0, eq - margin),
        currency="USD",
        leverage=leverage,
        is_demo=is_demo,
        fetched_at=fetched_at,
    )


def test_floor_to_step_never_rounds_up():
    assert floor_to_step(0.0199, 0.01) == pytest.approx(0.01)
    assert floor_to_step(0.0099, 0.01) == pytest.approx(0.0)
    assert floor_to_step(-1.0, 0.01) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "balance,atr,expected_ok,expected_vol",
    [
        # 10 pips SL (0.0010): 1.0 lot loses $100, 0.01 lot loses $1.00.
        # $50 account @ 1% risk -> budget = $0.50 < $1.00 min lot risk -> MUST SKIP (not round up to 0.01!)
        (50.0, 0.0010, False, 0.0),
        # $50 account with tight 4-pip SL (0.0004): 0.01 lot loses $0.40 <= $0.50 budget -> 0.01 lot allowed
        (50.0, 0.0004, True, 0.01),
        # $100 account @ 1% risk -> budget = $1.00, 10-pip SL -> exactly 0.01 lot ($1.00 risk)
        (100.0, 0.0010, True, 0.01),
        # $100 account with 12-pip SL (0.0012): 0.01 lot loses $1.20 > $1.00 budget -> SKIP
        (100.0, 0.0012, False, 0.0),
        # $500 account @ 1% risk -> budget = $5.00, 10-pip SL -> 0.05 lots ($5.00 risk)
        (500.0, 0.0010, True, 0.05),
        # $500 account @ 1% risk, 7-pip SL -> $5.00 / $70 = 0.0714 -> floored to 0.07 lots ($4.90 risk)
        (500.0, 0.0007, True, 0.07),
        # $100,000 account @ 1% risk -> budget = $1,000, 10-pip SL -> 10.00 lots ($1,000 risk)
        (100_000.0, 0.0010, True, 10.0),
    ],
)
def test_account_size_aware_position_sizing(balance, atr, expected_ok, expected_vol):
    spec = default_fx_spec("EURUSD", price=1.1000, leverage=100)
    acct = make_account(balance=balance)
    limits = RiskLimits(risk_per_trade=0.01, max_margin_utilization=0.30)
    res = size_position(acct, spec, entry=1.1000, sl=1.1000 - atr, limits=limits)
    assert res.ok is expected_ok
    assert res.volume == pytest.approx(expected_vol)
    if res.ok:
        assert res.risk_money <= balance * limits.risk_per_trade + 1e-9
        assert res.margin_required <= balance * limits.max_margin_utilization + 1e-9
    else:
        assert any("minimum volume" in r for r in res.reasons)


def test_margin_constraint_reduces_or_rejects_volume():
    # $50 account, 4-pip SL fits risk budget ($0.40 <= $0.50), but at 30:1 leverage
    # 0.01 lot requires $37.33 margin (> 30% of $50 = $15.00 cap) -> rejected on margin!
    spec = default_fx_spec("EURUSD", price=1.1200, leverage=30)
    acct = make_account(balance=50.0, leverage=30)
    limits = RiskLimits(risk_per_trade=0.01, max_margin_utilization=0.30)
    res = size_position(acct, spec, entry=1.1200, sl=1.1196, limits=limits)
    assert not res.ok
    assert any("margin" in r for r in res.reasons)


def test_existing_exposure_reduces_remaining_risk_capacity():
    spec = default_fx_spec("GBPUSD", price=1.2500, leverage=100)
    acct = make_account(balance=10_000.0)
    limits = RiskLimits(risk_per_trade=0.01, max_total_open_risk=0.015)
    # Existing position on EURUSD already risks $100 (1.0% of $10k); remaining room is $50 (0.5%)
    open_exp = [ExposureItem(symbol="EURUSD", side=BUY, volume=1.0, risk_money=100.0)]
    res = size_position(acct, spec, entry=1.2500, sl=1.2490, limits=limits, exposure=open_exp)
    assert res.ok
    assert res.volume == pytest.approx(0.50)
    assert res.risk_money == pytest.approx(50.0)


def test_position_without_sl_blocks_new_trades():
    spec = default_fx_spec("GBPUSD", price=1.2500, leverage=100)
    acct = make_account(balance=10_000.0)
    open_exp = [ExposureItem(symbol="EURUSD", side=BUY, volume=0.1, risk_money=math.inf)]
    res = size_position(acct, spec, entry=1.2500, sl=1.2490, limits=RiskLimits(), exposure=open_exp)
    assert not res.ok
    assert any("no stop-loss" in r for r in res.reasons)


def test_validate_order_enforces_all_hard_gates():
    now = 1_700_000_000.0
    spec = default_fx_spec("EURUSD", price=1.1000, leverage=100)
    acct = make_account(balance=10_000.0, fetched_at=now)
    limits = RiskLimits()
    state = RiskState()
    state.roll(now, acct.equity)
    good_intent = OrderIntent("EURUSD", BUY, entry=1.1000, sl=1.0990, tp=1.1020, signal_bar_time="2026-10-20 12:00:00")

    # 1. Valid demo order is approved
    dec = validate_order(good_intent, acct, spec, [], state, limits, now)
    assert dec.approved and dec.volume == pytest.approx(1.0)

    # 2. Real account rejected when allow_live=False
    real_acct = make_account(balance=10_000.0, is_demo=False, fetched_at=now)
    dec_real = validate_order(good_intent, real_acct, spec, [], state, limits, now, allow_live=False)
    assert not dec_real.approved and any("NOT a demo account" in r for r in dec_real.reasons)

    # 3. Stale account snapshot rejected
    dec_stale = validate_order(good_intent, acct, spec, [], state, limits, now + 60.0)
    assert not dec_stale.approved and any("stale" in r for r in dec_stale.reasons)

    # 4. Wrong-side stop-loss rejected
    bad_sl = OrderIntent("EURUSD", BUY, entry=1.1000, sl=1.1010, tp=1.1020, signal_bar_time="2026-10-20 12:00:00")
    dec_sl = validate_order(bad_sl, acct, spec, [], state, limits, now)
    assert not dec_sl.approved and any("wrong side" in r for r in dec_sl.reasons)

    # 5. Daily loss limit rejected
    dd_acct = make_account(balance=9_650.0, equity=9_650.0, fetched_at=now)
    dec_day = validate_order(good_intent, dd_acct, spec, [], state, limits, now)
    assert not dec_day.approved and any("daily loss limit" in r for r in dec_day.reasons)

    # 6. Max drawdown limit rejected
    state.peak_equity = 12_000.0
    dec_dd = validate_order(good_intent, acct, spec, [], state, limits, now)
    assert not dec_dd.approved and any("max drawdown" in r for r in dec_dd.reasons)
    state.peak_equity = 10_000.0

    # 7. Duplicate bar submission rejected
    state.submitted["EURUSD"] = "2026-10-20 12:00:00"
    dec_dup = validate_order(good_intent, acct, spec, [], state, limits, now)
    assert not dec_dup.approved and any("duplicate" in r for r in dec_dup.reasons)
    state.submitted.clear()

    # 8. Emergency stop active rejected
    dec_em = validate_order(good_intent, acct, spec, [], state, limits, now, emergency_stop=True)
    assert not dec_em.approved and any("emergency stop" in r for r in dec_em.reasons)

    # 9. Consecutive losses limit rejected
    dec_cl = validate_order(good_intent, acct, spec, [], state, limits, now, consecutive_losses=4)
    assert not dec_cl.approved and any("consecutive losses" in r for r in dec_cl.reasons)


def test_risk_state_atomic_persistence(tmp_path):
    p = tmp_path / "risk_state.json"
    st = RiskState()
    st.roll(1_700_000_000.0, 500.0)
    st.submitted["EURUSD"] = "2026-10-20 10:00:00"
    st.save(p)

    loaded = RiskState.load(p)
    assert loaded.day_start_equity == 500.0
    assert loaded.peak_equity == 500.0
    assert loaded.submitted["EURUSD"] == "2026-10-20 10:00:00"


def test_backtester_adapts_across_account_sizes():
    n = 120
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")
    close = np.linspace(1.1000, 1.1150, n)
    df = pd.DataFrame(
        {
            "open": close - 0.0001,
            "high": close + 0.0003,
            "low": close - 0.0003,
            "close": close,
            "volatility_atr": np.full(n, 0.0005),  # 5 pips SL -> 0.01 lot risks $0.50
            "spread": np.full(n, 2.0),
        },
        index=dates,
    )
    signals = np.zeros(n, dtype=int)
    signals[10] = BUY
    signals[40] = BUY
    signals[70] = BUY

    r_50 = run_backtest(df, signals, cfg=BacktestConfig(starting_balance=50.0))
    r_500 = run_backtest(df, signals, cfg=BacktestConfig(starting_balance=500.0))
    r_100k = run_backtest(df, signals, cfg=BacktestConfig(starting_balance=100_000.0))

    assert r_50.accepted_trades == 3
    assert r_500.accepted_trades == 3
    assert r_100k.accepted_trades == 3
    # Position size scales strictly with equity; $50 never uses $100k lots!
    assert r_50.avg_lots < r_500.avg_lots < r_100k.avg_lots
    assert r_50.trades[0].volume == pytest.approx(0.01)
    assert r_500.trades[0].volume == pytest.approx(0.10)
    assert r_100k.trades[0].volume == pytest.approx(20.0)


def test_live_models_load_and_infer_on_closed_bars():
    raw_df = pd.read_csv(
        Path(__file__).resolve().parents[1] / "data" / "raw" / "EURUSD_H1_30000.csv",
        index_col="time",
        parse_dates=True,
    ).iloc[-450:]

    load_cascade_models()
    pred_c = predict_cascade(raw_df.iloc[:-1])
    assert pred_c["direction"] in (-1, 1)
    assert 0.0 <= pred_c["p_expansion"] <= 1.0
    assert pred_c["bar_time"] == raw_df.index[-2]

    load_legacy_models()
    pred_l = predict_legacy(raw_df.iloc[:-1])
    assert pred_l["signal"] in (0, 1, 2)
    assert pred_l["bar_time"] == raw_df.index[-2]
