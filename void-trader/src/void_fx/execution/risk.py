"""Position sizing and broker order-parameter helpers.

calculate_position_size is kept for backward compatibility but now delegates to the
account-aware risk manager. Behaviour changes (defect fixes):
  * never rounds UP to volume_min (the old code did max(volume_min, ...), which made a
    small account risk far more than risk_pct) -> returns 0.0 (skip) instead;
  * returns 0.0 instead of a hard-coded 0.01 lots when symbol info is unavailable;
  * rounds DOWN to the volume step instead of to the nearest step.
The live runners no longer call this; they use execution.order_gate.OrderGate.
"""

import MetaTrader5 as mt5

from void_fx.config import RISK_PERCENTAGE
from void_fx.execution.risk_manager import InstrumentSpec, floor_to_step, loss_per_lot_from_ticks


def calculate_position_size(balance: float, atr: float, symbol: str, risk_pct: float = RISK_PERCENTAGE) -> float:
    """Volatility-adjusted size for a 1*ATR stop; 0.0 means 'do not trade'."""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None or not balance or balance <= 0 or not atr or atr <= 0:
        return 0.0
    spec = InstrumentSpec(
        symbol=symbol, digits=symbol_info.digits, point=symbol_info.point,
        tick_size=symbol_info.trade_tick_size, tick_value_loss=symbol_info.trade_tick_value,
        volume_min=symbol_info.volume_min, volume_max=symbol_info.volume_max,
        volume_step=symbol_info.volume_step, contract_size=symbol_info.trade_contract_size,
        margin_per_lot=1.0,
    )
    if spec.problems():
        return 0.0
    risk_per_lot = loss_per_lot_from_ticks(spec, 1.0 + atr, 1.0)
    lots = floor_to_step(min(balance * risk_pct / risk_per_lot, spec.volume_max), spec.volume_step)
    return lots if lots >= spec.volume_min else 0.0


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
