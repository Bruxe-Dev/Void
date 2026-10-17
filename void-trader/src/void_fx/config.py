"""
Central configuration. Every hard-coded constant from the original
live_forex_expert.py lives here, with identical values, so behaviour is unchanged.
"""

from pathlib import Path

# ------------------------------------------------------------------
# PATHS (absolute, so scripts work from any working directory)
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
LEGACY_MODELS_DIR = MODELS_DIR / "legacy"
CASCADE_MODELS_DIR = MODELS_DIR / "cascade"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# ------------------------------------------------------------------
# BROKER / TERMINAL
# ------------------------------------------------------------------
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
ACCOUNT_NUMBER = 5057177985
MAGIC_NUMBER = 202610

# ------------------------------------------------------------------
# STRATEGY
# ------------------------------------------------------------------
SYMBOL = "EURUSD"
SYMBOLS = ["EURUSD", "GBPUSD"]
LOOKBACK_WINDOW = 24          # LSTM sequence length (hours)
RISK_PERCENTAGE = 0.01        # 1% of balance per trade
ATR_PERIOD = 14
PT_MULT = 2.0                 # take-profit distance in ATR
SL_MULT = 1.0                 # stop-loss distance in ATR
MAX_HOLDING_BARS = 24         # vertical barrier / purge gap (hours)

# ------------------------------------------------------------------
# LIVE LOOP
# ------------------------------------------------------------------
BARS_TO_FETCH = 500           # >=169 closed bars needed by the regime filter
CYCLE_INTERVAL_SECONDS = 3600

# Regime filter mode. False = SHADOW (log only). See scripts/calibrate_regime.py:
# on 2021-2026 history the hard veto removed better-than-average legacy BUY signals.
REGIME_VETO_ENABLED = False

# ------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------
TRAINING_BARS = 30_000
LEGACY_FEATURE_COLS = ["log_return", "volatility_atr", "hour_cos", "hour_sin"]
