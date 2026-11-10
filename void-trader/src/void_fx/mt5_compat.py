"""Optional MetaTrader5 import: the official package only exists on Windows.

On Linux/Colab, `mt5` is a stand-in object: module-level constants such as
`mt5.TIMEFRAME_H1` resolve to their real numeric values so modules can be
imported (training, backtests, tests), while any *function* call raises a clear
error explaining that live trading needs the Windows MetaTrader 5 terminal.
"""

try:
    import MetaTrader5 as mt5  # type: ignore
    MT5_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on Linux only
    MT5_AVAILABLE = False

    _CONSTANTS = {
        "TIMEFRAME_H1": 16385,
        "TIMEFRAME_M1": 1,
        "TIMEFRAME_D1": 16408,
        "ORDER_TYPE_BUY": 0,
        "ORDER_TYPE_SELL": 1,
        "TRADE_ACTION_DEAL": 1,
        "ORDER_TIME_GTC": 0,
        "ORDER_FILLING_FOK": 0,
        "ORDER_FILLING_IOC": 1,
        "ORDER_FILLING_RETURN": 2,
        "TRADE_RETCODE_DONE": 10009,
        "DEAL_ENTRY_OUT": 1,
        "ACCOUNT_TRADE_MODE_DEMO": 0,
        "ACCOUNT_TRADE_MODE_REAL": 2,
    }

    class _MissingMT5:
        def __getattr__(self, name):
            if name in _CONSTANTS:
                return _CONSTANTS[name]
            if name.isupper():
                return 0

            def _unavailable(*_a, **_k):
                raise RuntimeError(
                    f"mt5.{name}() unavailable: the MetaTrader5 Python package is Windows-only. "
                    "Run the live bot on Windows (or Windows Python under Wine) with the MT5 terminal."
                )
            return _unavailable

    mt5 = _MissingMT5()
