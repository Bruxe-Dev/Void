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

# Cycle scheduling. True = run each cycle CYCLE_ALIGN_DELAY_SECONDS after the H1 bar
# closes (matches the training/backtest assumption of acting on the just-closed bar).
# False = original behaviour (sleep CYCLE_INTERVAL_SECONDS after each cycle, which drifts).
CYCLE_ALIGN_TO_BAR_CLOSE = True
CYCLE_ALIGN_DELAY_SECONDS = 15

# ------------------------------------------------------------------
# RISK MANAGEMENT / EXECUTION SAFETY (see execution/risk_manager.py)
# Conservative demo-testing defaults; none of these is "safe" in general.
# Sizing ALWAYS uses the live account equity/balance read from MT5 at order time;
# there is no default account size anywhere in the execution path.
# ------------------------------------------------------------------
RISK_LIMITS = dict(
    risk_per_trade=RISK_PERCENTAGE,   # 1% planned loss per trade (unchanged strategy setting)
    max_total_open_risk=0.02,         # all open positions + pending orders + new order
    max_daily_loss=0.03,              # vs equity at 00:00 UTC, realised + unrealised
    max_drawdown=0.10,                # vs peak equity -> persistent HALT, manual reset
    max_margin_utilization=0.30,      # (used + new margin) / equity
    max_positions_total=2,
    max_positions_per_symbol=1,       # same as the original one-trade-at-a-time rule
    max_consecutive_losses=4,         # per UTC day
    max_account_age_s=15.0,
)
COMMISSION_PER_LOT_ROUND_TURN = 0.0   # account currency; set to your broker's commission
# Demo-only by default. Live trading requires BOTH this flag AND the environment
# variable VOID_FX_ALLOW_LIVE=I_UNDERSTAND_REAL_MONEY_RISK. Do not change for testing.
ALLOW_LIVE_TRADING = False
STATE_DIR = PROJECT_ROOT / "state"
RISK_STATE_FILE = STATE_DIR / "risk_state.json"
EMERGENCY_STOP_FILE = PROJECT_ROOT / "STOP_TRADING"   # create this file to block all new orders

# ------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------
TRAINING_BARS = 30_000
LEGACY_FEATURE_COLS = ["log_return", "volatility_atr", "hour_cos", "hour_sin"]
