"""Position sizing and broker order-parameter helpers (logic unchanged from the original live script)."""

import MetaTrader5 as mt5

from void_fx.config import RISK_PERCENTAGE


def calculate_position_size(balance: float, atr: float, symbol: str, risk_pct: float = RISK_PERCENTAGE) -> float:
    """
    Computes volatility-adjusted position size based on cash risk budget and ATR stop distance.
    """
    cash_risk_budget = balance * risk_pct

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return 0.01

    trade_tick_size = symbol_info.trade_tick_size
    trade_tick_value = symbol_info.trade_tick_value
    volume_step = symbol_info.volume_step

    # 1 lot risk for 1 ATR movement: atr / tick_size * tick_value
    risk_per_lot = (atr / trade_tick_size) * trade_tick_value

    if risk_per_lot <= 0:
        return symbol_info.volume_min

    raw_lots = cash_risk_budget / risk_per_lot

    # Precision rounding according to broker volume step
    step_precision = len(str(volume_step).split('.')[1]) if '.' in str(volume_step) else 0
    final_lots = round(round(raw_lots / volume_step) * volume_step, step_precision)
    return max(symbol_info.volume_min, min(symbol_info.volume_max, final_lots))


def get_compatible_filling_mode(symbol_info) -> int:
    """
    Resolves the appropriate order execution filling mode supported by the broker.
    """
    if symbol_info.filling_mode & 1:
        return mt5.ORDER_FILLING_FOK
    elif symbol_info.filling_mode & 2:
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_RETURN
