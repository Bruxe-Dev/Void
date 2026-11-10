"""
Final order-submission boundary. The ONLY place in the live code that calls
mt5.order_send. Strategy/model code produces an OrderIntent (no volume); this gate
re-reads fresh account / instrument / exposure state, runs every hard risk check
(risk_manager.validate_order), sizes the order from current equity, and only then
submits. Rejections are logged with reasons; nothing is retried with larger size,
stops are never removed, and existing positions are never closed as a side effect.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from void_fx.mt5_compat import mt5

from void_fx import config
from void_fx.execution import mt5_gateway as gw
from void_fx.execution.risk_manager import (
    BUY, OrderDecision, OrderIntent, RiskLimits, RiskState, drawdown_breached, validate_order,
)
from void_fx.execution.risk import get_compatible_filling_mode

LIVE_ENV_VAR = "VOID_FX_ALLOW_LIVE"
LIVE_ENV_VALUE = "I_UNDERSTAND_REAL_MONEY_RISK"


def live_trading_allowed() -> bool:
    return bool(config.ALLOW_LIVE_TRADING) and os.environ.get(LIVE_ENV_VAR) == LIVE_ENV_VALUE


def emergency_stop_active() -> bool:
    return config.EMERGENCY_STOP_FILE.exists()


class OrderGate:
    def __init__(self, magic: int, limits: RiskLimits | None = None, dry_run: bool = False,
                 state_path=None, log=print):
        self.magic = magic
        self.limits = limits or RiskLimits(**config.RISK_LIMITS)
        self.dry_run = dry_run
        self.state_path = state_path or config.RISK_STATE_FILE
        self.log = log

    def _load_state(self) -> RiskState | None:
        try:
            return RiskState.load(self.state_path)
        except Exception as exc:   # corrupt state file -> fail closed
            self.log(f"[RISK] Cannot read risk state {self.state_path}: {exc!r}. Rejecting new orders.")
            return None

    def submit(self, intent: OrderIntent, deviation: int = 10) -> OrderDecision:
        state = self._load_state()
        if state is None:
            return OrderDecision(False, reasons=["risk state unreadable"])

        tick = mt5.symbol_info_tick(intent.symbol)
        account = gw.read_account()
        if account is not None:
            state.roll(time.time(), account.equity)
            if drawdown_breached(state, account.equity, self.limits) and not state.halted:
                state.halted = True
                state.halt_reason = (f"max drawdown {self.limits.max_drawdown * 100:.0f}% breached at "
                                     f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}; "
                                     f"delete 'halted' flag in {self.state_path.name} after review")
            state.save(self.state_path)

        price = None if tick is None else (tick.ask if intent.side == BUY else tick.bid)
        spec = gw.read_instrument(intent.symbol, intent.side, price) if price else None
        exposure = gw.read_exposure()
        day_start = datetime.strptime(state.day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() if state.day else time.time()
        losses = gw.consecutive_losses_today(self.magic, day_start)
        lpl = gw.loss_per_lot(intent.symbol, intent.side, intent.entry, intent.sl)
        spread_points = None if (tick is None or spec is None) else (tick.ask - tick.bid) / spec.point

        reasons_pre = []
        if tick is None:
            reasons_pre.append("no live tick")
        if losses is None:
            reasons_pre.append("deal history unavailable (cannot evaluate consecutive losses)")
        if lpl is None:
            reasons_pre.append("broker could not compute loss at stop-loss (order_calc_profit failed)")

        # Entry is the ask (BUY) / bid (SELL), so entry->SL already includes the spread.
        cost_per_lot = config.COMMISSION_PER_LOT_ROUND_TURN

        decision = validate_order(
            intent, account, spec, exposure, state, self.limits, time.time(),
            allow_live=live_trading_allowed(), consecutive_losses=losses or 0,
            cost_per_lot=cost_per_lot, loss_per_lot=lpl, spread_points=spread_points,
            emergency_stop=emergency_stop_active(),
        )
        if reasons_pre:
            decision = OrderDecision(False, sizing=decision.sizing, reasons=reasons_pre + decision.reasons)

        tag = f"[RISK] {intent.symbol} {'BUY' if intent.side == BUY else 'SELL'} bar={intent.signal_bar_time}"
        if account is not None:
            self.log(f"{tag} | equity={account.equity:,.2f} {account.currency} balance={account.balance:,.2f} "
                     f"free_margin={account.margin_free:,.2f} demo={account.is_demo}")
        self.log(f"{tag} | {decision.summary()}")
        if not decision.approved:
            return decision

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": intent.symbol,
            "volume": decision.volume,
            "type": mt5.ORDER_TYPE_BUY if intent.side == BUY else mt5.ORDER_TYPE_SELL,
            "price": intent.entry,
            "sl": intent.sl,
            "tp": intent.tp,
            "deviation": deviation,
            "magic": self.magic,
            "comment": intent.comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": get_compatible_filling_mode(mt5.symbol_info(intent.symbol)),
        }
        check = mt5.order_check(request)
        if check is None or check.retcode != 0:
            decision.approved = False
            decision.reasons.append(f"broker pre-check rejected: {check.comment if check else mt5.last_error()}")
            self.log(f"{tag} | REJECTED by order_check: {decision.reasons[-1]}")
            return decision

        if self.dry_run:
            self.log(f"{tag} | DRY RUN: order validated but NOT sent.")
            return decision

        # Record the submission BEFORE sending so a crash/restart cannot double-submit this bar.
        state.submitted[intent.symbol] = intent.signal_bar_time
        state.save(self.state_path)

        result = mt5.order_send(request)
        if result is not None and result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            self.log(f"{tag} | SENT: deal={result.deal} order={result.order} volume={result.volume}")
        else:
            decision.approved = False
            decision.reasons.append(
                f"order_send failed: code={getattr(result, 'retcode', 'NONE')} "
                f"comment={getattr(result, 'comment', mt5.last_error())} (not retried)")
            self.log(f"{tag} | {decision.reasons[-1]}")
        return decision
