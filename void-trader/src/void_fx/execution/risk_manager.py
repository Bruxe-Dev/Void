"""
Account-size-aware risk management (pure Python, no MetaTrader5 import).

This module is the single source of truth for:
  * position sizing from current equity, stop distance and instrument specs,
  * hard risk limits (per-trade, aggregate, daily loss, drawdown, margin,
    concurrent positions, consecutive losses, emergency halt),
  * the final pre-submission order validation.

It is deliberately broker-agnostic so the identical code path is used by the
live MT5 order gate (void_fx.execution.order_gate) and by the backtester
(void_fx.backtest.engine), and so it can be unit-tested deterministically.

Design rules
------------
* Never fall back to a default account size. Missing / stale / invalid account
  or instrument data -> reject the order.
* Volume is always rounded DOWN to the broker step. If the minimum volume would
  exceed the risk budget the trade is SKIPPED, never rounded up.
* Existing positions and pending orders consume risk capacity. A position or
  pending order without a stop-loss has unknown (unbounded) risk, so no new risk
  is added while one exists.
* A stop-loss caps the *planned* loss only. Gaps and slippage can produce larger
  realised losses; nothing here is a guarantee against loss.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

BUY, SELL = 1, -1


# ======================================================================
# DATA MODEL
# ======================================================================
@dataclass(frozen=True)
class AccountSnapshot:
    """Authoritative account state, read immediately before a decision."""
    balance: float
    equity: float
    margin: float            # margin already used
    margin_free: float
    currency: str
    leverage: int
    is_demo: bool
    fetched_at: float        # unix seconds (time.time()) when this snapshot was read

    def problems(self, now: float, max_age_s: float) -> list[str]:
        p = []
        for name in ("balance", "equity", "margin", "margin_free"):
            v = getattr(self, name)
            if v is None or not math.isfinite(v):
                p.append(f"account {name} is missing or non-finite")
        if not p:
            if self.equity <= 0 or self.balance <= 0:
                p.append(f"account balance/equity not positive (balance={self.balance:.2f}, equity={self.equity:.2f})")
            if self.margin < 0 or self.margin_free < 0:
                p.append("account margin values are negative")
        if not self.currency:
            p.append("account currency unknown")
        age = now - self.fetched_at
        if age < 0 or age > max_age_s:
            p.append(f"account snapshot is stale ({age:.1f}s old > {max_age_s:.0f}s)")
        return p


@dataclass(frozen=True)
class InstrumentSpec:
    """Contract specification. tick_value_loss is in ACCOUNT currency per tick per 1.0 lot."""
    symbol: str
    digits: int
    point: float
    tick_size: float
    tick_value_loss: float
    volume_min: float
    volume_max: float
    volume_step: float
    contract_size: float
    margin_per_lot: float    # account currency required for 1.0 lot at the current price
    stops_level_points: int = 0

    def problems(self) -> list[str]:
        p = []
        for name in ("point", "tick_size", "tick_value_loss", "volume_min", "volume_max",
                     "volume_step", "contract_size", "margin_per_lot"):
            v = getattr(self, name)
            if v is None or not math.isfinite(v) or v <= 0:
                p.append(f"{self.symbol}: instrument {name} invalid ({v})")
        if not p and self.volume_min > self.volume_max:
            p.append(f"{self.symbol}: volume_min > volume_max")
        return p


@dataclass(frozen=True)
class ExposureItem:
    """An open position or pending order that already consumes risk capacity."""
    symbol: str
    side: int
    volume: float
    risk_money: float        # loss in account currency if its SL is hit from now; math.inf if no SL
    kind: str = "position"   # "position" | "pending"
    magic: int = 0


@dataclass(frozen=True)
class OrderIntent:
    """What the strategy wants. Volume is NOT chosen by the strategy."""
    symbol: str
    side: int                # BUY (+1) or SELL (-1)
    entry: float
    sl: float
    tp: float | None
    signal_bar_time: str     # ISO time of the closed bar that produced the signal (idempotency key)
    comment: str = ""


@dataclass
class RiskLimits:
    """
    Conservative demo-testing defaults. None of these values is "safe" in general;
    they bound planned losses under normal fills only. See readme.md for rationale.
    """
    risk_per_trade: float = 0.01          # max planned loss per trade, fraction of equity basis
    max_total_open_risk: float = 0.02     # all open positions + pending orders + new trade
    max_daily_loss: float = 0.03          # vs equity at the start of the UTC day (realised + unrealised)
    max_drawdown: float = 0.10            # vs peak equity -> HALT (manual reset required)
    max_margin_utilization: float = 0.30  # (used margin + new margin) / equity
    max_positions_total: int = 2
    max_positions_per_symbol: int = 1
    max_consecutive_losses: int = 4       # per UTC day; pauses new entries until the next day
    max_account_age_s: float = 15.0
    max_spread_points: float | None = None

    def problems(self) -> list[str]:
        p = []
        for name in ("risk_per_trade", "max_total_open_risk", "max_daily_loss", "max_drawdown",
                     "max_margin_utilization"):
            v = getattr(self, name)
            if not (0 < v <= 1):
                p.append(f"limit {name}={v} must be in (0, 1]")
        if self.risk_per_trade > self.max_total_open_risk:
            p.append("risk_per_trade cannot exceed max_total_open_risk")
        return p


@dataclass
class SizingResult:
    ok: bool
    volume: float = 0.0
    risk_money: float = 0.0          # planned loss at SL incl. costs, account currency
    risk_fraction: float = 0.0       # risk_money / equity basis
    budget: float = 0.0
    loss_per_lot: float = 0.0
    margin_required: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class OrderDecision:
    approved: bool
    volume: float = 0.0
    sizing: SizingResult | None = None
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.approved and self.sizing:
            s = self.sizing
            return (f"APPROVED volume={self.volume} planned_loss={s.risk_money:.2f} "
                    f"({s.risk_fraction * 100:.2f}% of equity) margin={s.margin_required:.2f}")
        return "REJECTED: " + "; ".join(self.reasons)


# ======================================================================
# PERSISTENT RISK STATE (daily loss / drawdown / halt / de-duplication)
# ======================================================================
@dataclass
class RiskState:
    day: str = ""                       # UTC date of the current risk day (YYYY-MM-DD)
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    submitted: dict = field(default_factory=dict)   # "SYMBOL" -> signal_bar_time of the last submission

    @staticmethod
    def utc_day(now: float) -> str:
        return datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")

    def roll(self, now: float, equity: float) -> None:
        """Daily reset at 00:00 UTC; peak equity tracks the high-water mark."""
        day = self.utc_day(now)
        if day != self.day:
            self.day = day
            self.day_start_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "RiskState":
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))   # corrupt file -> raises (fail closed)
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise


# ======================================================================
# SIZING
# ======================================================================
def floor_to_step(volume: float, step: float) -> float:
    """Rounds DOWN to a multiple of `step` using decimal arithmetic (no float drift)."""
    if not (math.isfinite(volume) and volume > 0 and step > 0):
        return 0.0
    d_step = Decimal(str(step))
    units = (Decimal(str(volume)) / d_step).to_integral_value(rounding=ROUND_FLOOR)
    return float(units * d_step)


def loss_per_lot_from_ticks(spec: InstrumentSpec, entry: float, sl: float) -> float:
    """Monetary loss for 1.0 lot moving from entry to SL, using tick size / tick value."""
    ticks = abs(entry - sl) / spec.tick_size
    return ticks * spec.tick_value_loss


def size_position(
    account: AccountSnapshot,
    spec: InstrumentSpec,
    entry: float,
    sl: float,
    limits: RiskLimits,
    exposure: list[ExposureItem] | tuple = (),
    cost_per_lot: float = 0.0,
    loss_per_lot: float | None = None,
) -> SizingResult:
    """
    Largest broker-valid volume whose planned loss (SL distance + costs) fits within
    BOTH the per-trade budget and the remaining aggregate-risk capacity, and whose
    margin fits within the margin-utilisation limit. Never rounds up.

    loss_per_lot: optional authoritative loss for 1.0 lot entry->SL in account currency
                  (e.g. from mt5.order_calc_profit, which includes currency conversion).
    """
    res = SizingResult(ok=False)
    res.reasons += spec.problems() + limits.problems()
    if not all(math.isfinite(x) for x in (entry, sl, cost_per_lot)) or entry <= 0 or sl <= 0:
        res.reasons.append("entry/SL/cost not finite or not positive")
    if res.reasons:
        return res
    if account.equity <= 0 or account.balance <= 0:
        res.reasons.append("account equity/balance not positive")
        return res

    basis = min(account.balance, account.equity)   # conservative: never size off unrealised profit
    per_trade_budget = basis * limits.risk_per_trade

    open_risk = 0.0
    for e in exposure:
        if not math.isfinite(e.risk_money):
            res.reasons.append(f"existing {e.kind} on {e.symbol} has no stop-loss; aggregate risk unbounded")
            return res
        open_risk += max(0.0, e.risk_money)
    aggregate_room = basis * limits.max_total_open_risk - open_risk
    budget = min(per_trade_budget, aggregate_room)
    res.budget = max(0.0, budget)
    if budget <= 0:
        res.reasons.append(f"no aggregate risk capacity left (open risk {open_risk:.2f} >= "
                           f"{limits.max_total_open_risk * 100:.1f}% of {basis:.2f})")
        return res

    lpl = loss_per_lot if loss_per_lot is not None else loss_per_lot_from_ticks(spec, entry, sl)
    if not math.isfinite(lpl) or lpl <= 0:
        res.reasons.append(f"invalid loss-per-lot ({lpl}); stop distance zero or conversion failed")
        return res
    per_lot = lpl + max(0.0, cost_per_lot)
    res.loss_per_lot = per_lot

    vol = floor_to_step(min(budget / per_lot, spec.volume_max), spec.volume_step)

    # Margin constraint (can only REDUCE volume).
    margin_room = min(account.equity * limits.max_margin_utilization - account.margin, account.margin_free)
    vol_margin = floor_to_step(margin_room / spec.margin_per_lot, spec.volume_step) if margin_room > 0 else 0.0
    margin_limited = vol_margin < vol
    vol = min(vol, vol_margin)

    if vol < spec.volume_min - 1e-12:
        min_risk = spec.volume_min * per_lot
        min_margin = spec.volume_min * spec.margin_per_lot
        if margin_limited and min_margin > max(margin_room, 0.0):
            res.reasons.append(
                f"minimum volume {spec.volume_min} needs margin {min_margin:.2f} {account.currency} > allowed "
                f"{max(margin_room, 0.0):.2f} ({limits.max_margin_utilization * 100:.0f}% utilisation cap)")
        else:
            res.reasons.append(
                f"minimum volume {spec.volume_min} would risk {min_risk:.2f} {account.currency} "
                f"({min_risk / basis * 100:.2f}% of equity) > budget {budget:.2f} "
                f"({budget / basis * 100:.2f}%); trade skipped, not rounded up")
        return res

    res.ok = True
    res.volume = vol
    res.risk_money = vol * per_lot
    res.risk_fraction = res.risk_money / basis
    res.margin_required = vol * spec.margin_per_lot
    return res


# ======================================================================
# FINAL ORDER VALIDATION (called at the submission boundary)
# ======================================================================
def validate_order(
    intent: OrderIntent,
    account: AccountSnapshot | None,
    spec: InstrumentSpec | None,
    exposure: list[ExposureItem] | None,
    state: RiskState,
    limits: RiskLimits,
    now: float,
    *,
    allow_live: bool = False,
    consecutive_losses: int = 0,
    cost_per_lot: float = 0.0,
    loss_per_lot: float | None = None,
    spread_points: float | None = None,
    emergency_stop: bool = False,
) -> OrderDecision:
    """Runs every hard check. Any failure -> rejected with explicit reasons. Never mutates `state`."""
    reasons: list[str] = []

    # 0. Emergency halt / environment
    if emergency_stop:
        reasons.append("emergency stop is active")
    if state.halted:
        reasons.append(f"trading halted: {state.halt_reason or 'manual reset required'}")
    if account is None:
        return OrderDecision(False, reasons=reasons + ["account information unavailable"])
    if not account.is_demo and not allow_live:
        reasons.append("account is NOT a demo account and live trading is not explicitly enabled")

    # 1. Account freshness / validity
    reasons += account.problems(now, limits.max_account_age_s)
    # 2. Instrument
    if spec is None:
        return OrderDecision(False, reasons=reasons + [f"instrument spec for {intent.symbol} unavailable"])
    reasons += spec.problems()
    if exposure is None:
        return OrderDecision(False, reasons=reasons + ["open positions / pending orders could not be read"])

    # 4. Stop-loss validity
    if intent.side not in (BUY, SELL):
        reasons.append(f"invalid side {intent.side}")
    else:
        if not all(math.isfinite(x) and x > 0 for x in (intent.entry, intent.sl)):
            reasons.append("entry or stop-loss missing / non-finite")
        elif (intent.side == BUY and not intent.sl < intent.entry) or (intent.side == SELL and not intent.sl > intent.entry):
            reasons.append("stop-loss is on the wrong side of the entry price")
        elif spec and spec.point > 0 and abs(intent.entry - intent.sl) < max(spec.stops_level_points, 1) * spec.point:
            reasons.append("stop-loss closer than the broker minimum stop level")
        if intent.tp is not None and math.isfinite(intent.tp):
            if (intent.side == BUY and not intent.tp > intent.entry) or (intent.side == SELL and not intent.tp < intent.entry):
                reasons.append("take-profit is on the wrong side of the entry price")

    if limits.max_spread_points is not None and spread_points is not None and spread_points > limits.max_spread_points:
        reasons.append(f"spread {spread_points:.1f} points > limit {limits.max_spread_points:.1f}")

    # 8. Daily loss / drawdown / consecutive losses (use current equity = realised + unrealised)
    if account.equity > 0 and state.day_start_equity > 0:
        day_loss = (state.day_start_equity - account.equity) / state.day_start_equity
        if day_loss >= limits.max_daily_loss:
            reasons.append(f"daily loss limit reached ({day_loss * 100:.2f}% >= {limits.max_daily_loss * 100:.1f}%)")
    if account.equity > 0 and state.peak_equity > 0:
        dd = (state.peak_equity - account.equity) / state.peak_equity
        if dd >= limits.max_drawdown:
            reasons.append(f"max drawdown reached ({dd * 100:.2f}% >= {limits.max_drawdown * 100:.1f}%)")
    if consecutive_losses >= limits.max_consecutive_losses:
        reasons.append(f"{consecutive_losses} consecutive losses today (limit {limits.max_consecutive_losses})")

    # 9. Duplicates / concurrency
    if state.submitted.get(intent.symbol) == intent.signal_bar_time:
        reasons.append(f"duplicate: an order for {intent.symbol} was already submitted for bar {intent.signal_bar_time}")
    positions = [e for e in exposure if e.kind == "position"]
    if len(positions) >= limits.max_positions_total:
        reasons.append(f"{len(positions)} open positions >= limit {limits.max_positions_total}")
    same = [e for e in exposure if e.symbol == intent.symbol]
    if len(same) >= limits.max_positions_per_symbol:
        reasons.append(f"{len(same)} open position(s)/order(s) on {intent.symbol} >= limit {limits.max_positions_per_symbol}")

    if reasons:
        return OrderDecision(False, reasons=reasons)

    # 3/5/6/7. Volume, per-trade risk, aggregate risk, margin
    sizing = size_position(account, spec, intent.entry, intent.sl, limits, exposure,
                           cost_per_lot=cost_per_lot, loss_per_lot=loss_per_lot)
    if not sizing.ok:
        return OrderDecision(False, sizing=sizing, reasons=sizing.reasons)
    return OrderDecision(True, volume=sizing.volume, sizing=sizing)


def drawdown_breached(state: RiskState, equity: float, limits: RiskLimits) -> bool:
    return state.peak_equity > 0 and (state.peak_equity - equity) / state.peak_equity >= limits.max_drawdown
