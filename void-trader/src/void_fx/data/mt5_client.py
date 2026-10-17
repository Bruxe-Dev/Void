"""MetaTrader 5 connectivity and bar retrieval."""

import json
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from void_fx.config import ACCOUNT_NUMBER, MT5_PATH, RAW_DATA_DIR, SYMBOL


def connect_to_mt5(symbol: str = SYMBOL) -> bool:
    """
    Initializes MT5 connection and verifies account authorization.
    """
    if not mt5.initialize(path=MT5_PATH):
        print(f"[!] MT5 Initialization Failed: {mt5.last_error()}")
        return False

    account_info = mt5.account_info()
    if account_info is None:
        print(f"[!] Unable to retrieve account info: {mt5.last_error()}")
        return False

    if account_info.login != ACCOUNT_NUMBER:
        print(f"[!] Warning: Active account is {account_info.login}, configured account is {ACCOUNT_NUMBER}.")

    # Verify terminal Algo Trading permission
    terminal_info = mt5.terminal_info()
    if terminal_info and not terminal_info.trade_allowed:
        print("[!] Note: 'Algo Trading' is currently DISABLED in MT5 GUI (Tools -> Options -> Expert Advisors -> Allow Algo Trading).")
        print("    Ensure the 'Algo Trading' button in MT5 toolbar is turned ON (green) for live trade execution.")

    # Ensure symbol is active in Market Watch
    if not mt5.symbol_select(symbol, True):
        print(f"[!] Failed to select {symbol} in Market Watch: {mt5.last_error()}")
        return False

    return True


def rates_to_frame(raw_rates) -> pd.DataFrame:
    """Converts the numpy record array from copy_rates_* into a time-indexed DataFrame."""
    df = pd.DataFrame(raw_rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df


def fetch_history(symbol: str, num_bars: int, cache_dir: Path = RAW_DATA_DIR, refresh: bool = False):
    """
    Downloads CLOSED H1 bars (start_pos=1 skips the forming candle) for training.
    Cached as CSV so experiments are reproducible and offline-friendly.

    Returns (DataFrame[open, high, low, close, tick_volume, spread], point_size).
    """
    cache = Path(cache_dir) / f"{symbol}_H1_{num_bars}.csv"
    meta = Path(cache_dir) / f"{symbol}_meta.json"
    if cache.exists() and meta.exists() and not refresh:
        df = pd.read_csv(cache, index_col="time", parse_dates=True)
        return df, json.loads(meta.read_text())["point"]

    if not mt5.initialize(path=MT5_PATH):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 1, num_bars)
        info = mt5.symbol_info(symbol)
        if rates is None or info is None:
            raise RuntimeError(f"No data for {symbol}: {mt5.last_error()}")
        point = info.point
    finally:
        mt5.shutdown()

    df = rates_to_frame(rates)[["open", "high", "low", "close", "tick_volume", "spread"]]
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    meta.write_text(json.dumps({"point": point}))
    return df, point
