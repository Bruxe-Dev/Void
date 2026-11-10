"""
Read-only MetaTrader 5 adapters that convert terminal state into the broker-agnostic
risk_manager data model. Nothing in this module sends or modifies orders.

Every function returns None when the terminal returns None / invalid data, so callers
reject the order instead of silently assuming defaults.
"""

from __future__ import annotations

import math
import time

from void_fx.mt5_compat import mt5

from void_fx.execution.risk_manager import BUY, SELL, AccountSnapshot, ExposureItem, InstrumentSpec


def read_account() -> AccountSnapshot | None:
    a = mt5.account_info()
    if a is None:
        return None
    return AccountSnapshot(
        balance=float(a.balance), equity=float(a.equity), margin=float(a.margin),
        margin_free=float(a.margin_free), currency=str(a.currency or ""), leverage=int(a.leverage),
        is_demo=(a.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO), fetched_at=time.time(),
    )


def read_instrument(symbol: str, side: int, price: float) -> InstrumentSpec | None:
    info = mt5.symbol_info(symbol)
    if info is None or price is None or not math.isfinite(price) or price <= 0:
        return None
    order_type = mt5.ORDER_TYPE_BUY if side == BUY else mt5.ORDER_TYPE_SELL
    margin = mt5.order_calc_margin(order_type, symbol, 1.0, price)
    tick_value_loss = getattr(info, "trade_tick_value_loss", 0.0) or info.trade_tick_value
    return InstrumentSpec(
        symbol=symbol, digits=int(info.digits), point=float(info.point),
        tick_size=float(info.trade_tick_size), tick_value_loss=float(tick_value_loss),
        volume_min=float(info.volume_min), volume_max=float(info.volume_max),
        volume_step=float(info.volume_step), contract_size=float(info.trade_contract_size),
        margin_per_lot=float(margin) if margin is not None else float("nan"),
        stops_level_points=int(info.trade_stops_level),
    )


def loss_per_lot(symbol: str, side: int, entry: float, sl: float) -> float | None:
    """Authoritative loss of 1.0 lot from entry to SL in account currency (includes FX conversion)."""
    order_type = mt5.ORDER_TYPE_BUY if side == BUY else mt5.ORDER_TYPE_SELL
    p = mt5.order_calc_profit(order_type, symbol, 1.0, entry, sl)
    if p is None or not math.isfinite(p):
        return None
    return abs(min(p, 0.0)) if p < 0 else None   # profit >= 0 at the SL means the SL is misplaced


def _risk_from(order_type_buy: bool, symbol: str, volume: float, price_now: float, sl: float) -> float:
    if not sl:
        return math.inf
    ot = mt5.ORDER_TYPE_BUY if order_type_buy else mt5.ORDER_TYPE_SELL
    p = mt5.order_calc_profit(ot, symbol, volume, price_now, sl)
    if p is None or not math.isfinite(p):
        return math.inf   # unknown -> treat as unbounded (fail closed)
    return max(0.0, -p)


def read_exposure() -> list[ExposureItem] | None:
    """All open positions AND pending orders on the account (any magic: they all consume capacity)."""
    positions = mt5.positions_get()
    orders = mt5.orders_get()
    if positions is None or orders is None:
        return None   # terminal error: an empty result is (), never None
    items: list[ExposureItem] = []
    for p in positions:
        is_buy = p.type == mt5.POSITION_TYPE_BUY
        items.append(ExposureItem(p.symbol, BUY if is_buy else SELL, float(p.volume),
                                  _risk_from(is_buy, p.symbol, p.volume, p.price_current, p.sl),
                                  "position", int(p.magic)))
    buy_types = {mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_BUY_STOP_LIMIT}
    for o in orders:
        is_buy = o.type in buy_types
        items.append(ExposureItem(o.symbol, BUY if is_buy else SELL, float(o.volume_current),
                                  _risk_from(is_buy, o.symbol, o.volume_current, o.price_open, o.sl),
                                  "pending", int(o.magic)))
    return items


def consecutive_losses_today(magic: int, day_start_epoch: float) -> int | None:
    """Consecutive losing closed deals (most recent first) for this bot since 00:00 UTC."""
    deals = mt5.history_deals_get(int(day_start_epoch), int(time.time()) + 86400)
    if deals is None:
        return None
    closes = [d for d in deals if d.magic == magic and d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY)]
    closes.sort(key=lambda d: d.time_msc, reverse=True)
    n = 0
    for d in closes:
        if d.profit + d.commission + d.swap < 0:
            n += 1
        else:
            break
    return n
