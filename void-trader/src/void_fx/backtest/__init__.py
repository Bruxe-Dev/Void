"""Account-size-aware historical backtesting engine for Void FX."""

from void_fx.backtest.engine import (
    BacktestConfig,
    BacktestPosition,
    BacktestResult,
    ClosedTrade,
    default_fx_spec,
    run_backtest,
)

__all__ = [
    "BacktestConfig",
    "BacktestPosition",
    "BacktestResult",
    "ClosedTrade",
    "default_fx_spec",
    "run_backtest",
]
