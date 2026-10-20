"""
Account-size-aware historical backtest and simulation engine.

Uses the exact same `validate_order` and `size_position` logic from
`void_fx.execution.risk_manager` as the live `OrderGate`.

Tracks at every bar:
  * starting_balance, balance, equity, unrealized_pnl, used_margin, free_margin
  * open positions, mark-to-market equity curve, peak-to-trough drawdown
  * spread, commission, slippage, and daily rollover swap costs
  * daily loss limit (00:00 UTC reset), max drawdown halt, consecutive loss limit
  * accepted trades vs rejected signals (with reason counts)
  * raw signal R-multiples (separating pure model quality from account sizing)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from void_fx.config import MAX_HOLDING_BARS, PT_MULT, RISK_LIMITS, SL_MULT
from void_fx.execution.risk_manager import (
    BUY,
    SELL,
    AccountSnapshot,
    ExposureItem,
    InstrumentSpec,
    OrderIntent,
    RiskLimits,
    RiskState,
    drawdown_breached,
    loss_per_lot_from_ticks,
    validate_order,
)


def default_fx_spec(
    symbol: str = "EURUSD",
    price: float = 1.1200,
    leverage: int = 100,
    point: float = 1e-5,
    digits: int = 5,
    contract_size: float = 100_000.0,
    volume_min: float = 0.01,
    volume_max: float = 500.0,
    volume_step: float = 0.01,
    tick_value_loss: float = 1.0,
    stops_level_points: int = 0,
) -> InstrumentSpec:
    """Builds a standard FX InstrumentSpec (matches live MT5 EURUSD/GBPUSD specs)."""
    margin_per_lot = (contract_size * price) / max(leverage, 1)
    return InstrumentSpec(
        symbol=symbol,
        digits=digits,
        point=point,
        tick_size=point,
        tick_value_loss=tick_value_loss,
        volume_min=volume_min,
        volume_max=volume_max,
        volume_step=volume_step,
        contract_size=contract_size,
        margin_per_lot=margin_per_lot,
        stops_level_points=stops_level_points,
    )


@dataclass
class BacktestConfig:
    starting_balance: float = 100_000.0
    leverage: int = 100
    currency: str = "USD"
    pt_mult: float = PT_MULT
    sl_mult: float = SL_MULT
    max_holding_bars: int = MAX_HOLDING_BARS
    max_spread_pips: float = 2.5
    default_spread_points: float = 2.0      # used when bar spread == 0
    slippage_points: float = 0.0
    commission_per_lot: float = 0.0         # round-turn in account currency
    swap_per_lot_per_day: float = 0.0       # daily financing cost per 1.0 lot
    use_intrabar_hl: bool = False           # False = close-based barriers (matches training labels)
    limits: RiskLimits = field(default_factory=lambda: RiskLimits(**RISK_LIMITS))


@dataclass
class BacktestPosition:
    symbol: str
    side: int
    volume: float
    entry_price: float
    sl: float
    tp: float
    entry_bar_idx: int
    entry_time: pd.Timestamp
    planned_risk_money: float
    planned_risk_pct: float
    margin_used: float
    commission_paid: float = 0.0
    swap_accum: float = 0.0
    last_day: str = ""


@dataclass
class ClosedTrade:
    symbol: str
    side: int
    volume: float
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    exit_reason: str            # "TP" | "SL" | "EXPIRY"
    gross_pnl: float
    costs: float
    net_pnl: float
    realized_r: float
    planned_risk_money: float
    planned_risk_pct: float
    bars_held: int


@dataclass
class BacktestResult:
    starting_balance: float
    ending_balance: float
    ending_equity: float
    net_profit: float
    return_pct: float
    max_drawdown_pct: float
    max_drawdown_money: float
    peak_Margin_utilization_pct: float
    total_signals: int
    accepted_trades: int
    rejected_orders: int
    rejection_reasons: dict[str, int]
    win_rate: float
    profit_factor: float
    avg_lots: float
    min_lots: float
    max_lots: float
    avg_planned_risk_money: float
    avg_planned_risk_pct: float
    signal_only_mean_r: float      # pure model/strategy edge (net of spread in R), independent of account size
    signal_only_win_rate: float
    halted: bool
    halt_reason: str
    trades: list[ClosedTrade] = field(default_factory=list)


def _evaluate_raw_signal_r(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    spread_pts: np.ndarray,
    point: float,
    idx: int,
    side: int,
    cfg: BacktestConfig,
) -> tuple[float, bool] | None:
    """Computes the R-multiple of a signal at bar `idx` independent of account size."""
    n = len(close)
    if idx + 1 >= n:
        return None
    a = atr[idx]
    if not np.isfinite(a) or a <= 0:
        return None
    sp_price = (spread_pts[idx] + cfg.slippage_points) * point
    cost_r = sp_price / a
    c0 = close[idx]
    end = min(idx + cfg.max_holding_bars, n - 1)
    for j in range(idx + 1, end + 1):
        if cfg.use_intrabar_hl:
            fav = (high[j] - c0) / a if side == BUY else (c0 - low[j]) / a
            adv = (c0 - low[j]) / a if side == BUY else (high[j] - c0) / a
            if adv >= cfg.sl_mult:
                return -1.0 - cost_r, False
            if fav >= cfg.pt_mult:
                return (cfg.pt_mult / cfg.sl_mult) - cost_r, True
        else:
            move = side * (close[j] - c0) / a
            if move <= -cfg.sl_mult:
                return -1.0 - cost_r, False
            if move >= cfg.pt_mult:
                return (cfg.pt_mult / cfg.sl_mult) - cost_r, True
    move_end = side * (close[end] - c0) / a
    r_net = (move_end / cfg.sl_mult) - cost_r
    return float(r_net), r_net > 0


def run_backtest(
    df: pd.DataFrame,
    signals: Sequence[int],
    symbol: str = "EURUSD",
    point: float = 1e-5,
    cfg: BacktestConfig | None = None,
    spec_override: InstrumentSpec | None = None,
) -> BacktestResult:
    """
    Runs a chronological, account-aware simulation over `df`.

    Parameters
    ----------
    df : DataFrame indexed by timestamp with ['open', 'high', 'low', 'close', 'volatility_atr']
         and optional 'spread'.
    signals : sequence of int (len == len(df)), where +1 = BUY, -1 = SELL, 0 = flat.
    """
    cfg = cfg or BacktestConfig()
    if len(df) != len(signals):
        raise ValueError(f"df ({len(df)}) and signals ({len(signals)}) must have equal length")

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr = df["volatility_atr"].to_numpy(dtype=float)
    raw_spread = df["spread"].to_numpy(dtype=float) if "spread" in df.columns else np.zeros(len(df))
    spread_pts = np.where(raw_spread > 0, raw_spread, cfg.default_spread_points)
    times = df.index

    balance = float(cfg.starting_balance)
    peak_equity = balance
    max_dd_pct = 0.0
    max_dd_money = 0.0
    peak_margin_util = 0.0

    state = RiskState()
    open_pos: BacktestPosition | None = None
    trades: list[ClosedTrade] = []
    rejections: Counter = Counter()
    total_signals = 0
    consec_losses_today = 0
    current_utc_day = ""

    raw_r_list: list[float] = []
    raw_wins: list[bool] = []

    for i in range(len(df)):
        ts: pd.Timestamp = times[i]
        now_epoch = float(ts.timestamp())
        day_str = ts.strftime("%Y-%m-%d")
        if day_str != current_utc_day:
            current_utc_day = day_str
            consec_losses_today = 0

        mid = close[i]
        sp = spread_pts[i] * point
        bid = mid - 0.5 * sp
        ask = mid + 0.5 * sp

        spec = spec_override or default_fx_spec(symbol=symbol, price=mid, leverage=cfg.leverage, point=point)

        # 1. Update open position (swap + exit checks + unrealized P&L)
        if open_pos is not None:
            if day_str != open_pos.last_day:
                open_pos.swap_accum += cfg.swap_per_lot_per_day * open_pos.volume
                open_pos.last_day = day_str

            bars_held = i - open_pos.entry_bar_idx
            exit_reason = None
            exit_price = None

            if cfg.use_intrabar_hl:
                if open_pos.side == BUY:
                    if low[i] - 0.5 * sp <= open_pos.sl:
                        exit_reason, exit_price = "SL", open_pos.sl - cfg.slippage_points * point
                    elif high[i] - 0.5 * sp >= open_pos.tp:
                        exit_reason, exit_price = "TP", open_pos.tp
                else:
                    if high[i] + 0.5 * sp >= open_pos.sl:
                        exit_reason, exit_price = "SL", open_pos.sl + cfg.slippage_points * point
                    elif low[i] + 0.5 * sp <= open_pos.tp:
                        exit_reason, exit_price = "TP", open_pos.tp
            else:
                # Close-based barrier evaluation (matches training triple-barrier convention)
                if open_pos.side == BUY:
                    if mid <= open_pos.sl:
                        exit_reason, exit_price = "SL", bid - cfg.slippage_points * point
                    elif mid >= open_pos.tp:
                        exit_reason, exit_price = "TP", bid
                else:
                    if mid >= open_pos.sl:
                        exit_reason, exit_price = "SL", ask + cfg.slippage_points * point
                    elif mid <= open_pos.tp:
                        exit_reason, exit_price = "TP", ask

            if exit_reason is None and bars_held >= cfg.max_holding_bars:
                exit_reason = "EXPIRY"
                exit_price = (bid if open_pos.side == BUY else ask) - (
                    open_pos.side * cfg.slippage_points * point
                )

            if exit_reason is not None and exit_price is not None:
                price_diff = (exit_price - open_pos.entry_price) * open_pos.side
                gross_pnl = (price_diff / spec.tick_size) * spec.tick_value_loss * open_pos.volume
                costs = open_pos.commission_paid + open_pos.swap_accum
                net_pnl = gross_pnl - costs
                balance += net_pnl
                r_mult = net_pnl / open_pos.planned_risk_money if open_pos.planned_risk_money > 0 else 0.0
                trades.append(
                    ClosedTrade(
                        symbol=symbol,
                        side=open_pos.side,
                        volume=open_pos.volume,
                        entry_time=str(open_pos.entry_time),
                        exit_time=str(ts),
                        entry_price=open_pos.entry_price,
                        exit_price=float(exit_price),
                        sl=open_pos.sl,
                        tp=open_pos.tp,
                        exit_reason=exit_reason,
                        gross_pnl=float(gross_pnl),
                        costs=float(costs),
                        net_pnl=float(net_pnl),
                        realized_r=float(r_mult),
                        planned_risk_money=open_pos.planned_risk_money,
                        planned_risk_pct=open_pos.planned_risk_pct,
                        bars_held=bars_held,
                    )
                )
                if net_pnl < 0:
                    consec_losses_today += 1
                else:
                    consec_losses_today = 0
                open_pos = None

        # Mark-to-market equity
        unrealized = 0.0
        used_margin = 0.0
        exposure: list[ExposureItem] = []
        if open_pos is not None:
            mark_price = bid if open_pos.side == BUY else ask
            unrealized = (
                ((mark_price - open_pos.entry_price) * open_pos.side / spec.tick_size)
                * spec.tick_value_loss
                * open_pos.volume
                - open_pos.swap_accum
            )
            used_margin = open_pos.margin_used
            rem_loss_per_lot = loss_per_lot_from_ticks(spec, mark_price, open_pos.sl)
            exposure.append(
                ExposureItem(
                    symbol=open_pos.symbol,
                    side=open_pos.side,
                    volume=open_pos.volume,
                    risk_money=rem_loss_per_lot * open_pos.volume,
                    kind="position",
                )
            )

        equity = balance + unrealized
        free_margin = max(0.0, equity - used_margin)
        if equity > 0:
            peak_margin_util = max(peak_margin_util, (used_margin / equity) * 100.0)

        state.roll(now_epoch, equity)
        if equity > peak_equity:
            peak_equity = equity
        if peak_equity > 0:
            dd_money = peak_equity - equity
            dd_pct = (dd_money / peak_equity) * 100.0
            if dd_money > max_dd_money:
                max_dd_money = dd_money
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
        if drawdown_breached(state, equity, cfg.limits) and not state.halted:
            state.halted = True
            state.halt_reason = f"max drawdown {cfg.limits.max_drawdown * 100:.1f}% reached at {ts}"

        # 2. Process new signal
        sig = int(signals[i])
        if sig not in (BUY, SELL) or i + 1 >= len(df):
            continue

        total_signals += 1
        raw_eval = _evaluate_raw_signal_r(close, high, low, atr, spread_pts, point, i, sig, cfg)
        if raw_eval is not None:
            raw_r_list.append(raw_eval[0])
            raw_wins.append(raw_eval[1])

        if open_pos is not None:
            rejections["position_already_open"] += 1
            continue

        spread_pips = spread_pts[i] / 10.0
        if spread_pips > cfg.max_spread_pips:
            rejections[f"spread_shield ({spread_pips:.1f}p > {cfg.max_spread_pips}p)"] += 1
            continue

        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            rejections["invalid_atr"] += 1
            continue

        slip = cfg.slippage_points * point
        if sig == BUY:
            entry_price = round(ask + slip, spec.digits)
            sl_price = round(entry_price - a * cfg.sl_mult, spec.digits)
            tp_price = round(entry_price + a * cfg.pt_mult, spec.digits)
        else:
            entry_price = round(bid - slip, spec.digits)
            sl_price = round(entry_price + a * cfg.sl_mult, spec.digits)
            tp_price = round(entry_price - a * cfg.pt_mult, spec.digits)

        account = AccountSnapshot(
            balance=balance,
            equity=equity,
            margin=used_margin,
            margin_free=free_margin,
            currency=cfg.currency,
            leverage=cfg.leverage,
            is_demo=True,
            fetched_at=now_epoch,
        )
        intent = OrderIntent(
            symbol=symbol,
            side=sig,
            entry=entry_price,
            sl=sl_price,
            tp=tp_price,
            signal_bar_time=str(ts),
            comment="Backtest",
        )
        decision = validate_order(
            intent,
            account,
            spec,
            exposure,
            state,
            cfg.limits,
            now_epoch,
            allow_live=False,
            consecutive_losses=consec_losses_today,
            cost_per_lot=cfg.commission_per_lot,
            spread_points=float(spread_pts[i]),
        )
        if not decision.approved or decision.sizing is None:
            raw_r = decision.reasons[0] if decision.reasons else "unknown"
            if raw_r.startswith("minimum volume") and "would risk" in raw_r:
                reason_key = "minimum_volume_exceeds_risk_budget"
            elif raw_r.startswith("minimum volume") and "needs margin" in raw_r:
                reason_key = "minimum_volume_exceeds_margin_limit"
            elif raw_r.startswith("trading halted:"):
                reason_key = "halted_max_drawdown"
            else:
                reason_key = raw_r.split("(")[0].strip()
            rejections[reason_key] += 1
            continue

        state.submitted[symbol] = str(ts)
        s = decision.sizing
        open_pos = BacktestPosition(
            symbol=symbol,
            side=sig,
            volume=decision.volume,
            entry_price=entry_price,
            sl=sl_price,
            tp=tp_price,
            entry_bar_idx=i,
            entry_time=ts,
            planned_risk_money=s.risk_money,
            planned_risk_pct=s.risk_fraction * 100.0,
            margin_used=s.margin_required,
            commission_paid=cfg.commission_per_lot * decision.volume,
            last_day=day_str,
        )
        if equity > 0:
            peak_margin_util = max(peak_margin_util, (open_pos.margin_used / equity) * 100.0)

    # Final stats
    n_trades = len(trades)
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    gross_win = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    lots = [t.volume for t in trades]
    risk_m = [t.planned_risk_money for t in trades]
    risk_p = [t.planned_risk_pct for t in trades]

    return BacktestResult(
        starting_balance=cfg.starting_balance,
        ending_balance=round(balance, 2),
        ending_equity=round(balance, 2),
        net_profit=round(balance - cfg.starting_balance, 2),
        return_pct=round(((balance - cfg.starting_balance) / cfg.starting_balance) * 100.0, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
        max_drawdown_money=round(max_dd_money, 2),
        peak_Margin_utilization_pct=round(peak_margin_util, 2),
        total_signals=total_signals,
        accepted_trades=n_trades,
        rejected_orders=int(sum(rejections.values())),
        rejection_reasons=dict(rejections),
        win_rate=round((len(wins) / n_trades) * 100.0, 2) if n_trades else 0.0,
        profit_factor=round(pf, 3) if np.isfinite(pf) else pf,
        avg_lots=round(float(np.mean(lots)), 4) if lots else 0.0,
        min_lots=round(float(np.min(lots)), 4) if lots else 0.0,
        max_lots=round(float(np.max(lots)), 4) if lots else 0.0,
        avg_planned_risk_money=round(float(np.mean(risk_m)), 2) if risk_m else 0.0,
        avg_planned_risk_pct=round(float(np.mean(risk_p)), 3) if risk_p else 0.0,
        signal_only_mean_r=round(float(np.mean(raw_r_list)), 4) if raw_r_list else 0.0,
        signal_only_win_rate=round(float(np.mean(raw_wins)) * 100.0, 2) if raw_wins else 0.0,
        halted=state.halted,
        halt_reason=state.halt_reason,
        trades=trades,
    )
