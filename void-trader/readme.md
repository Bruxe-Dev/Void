# Void FX: Hierarchical Stacked-Ensemble Algorithmic Forex Trading

Fully automated algorithmic Forex trading robot interfacing with MetaTrader 5 (MT5) on H1 candle frequency targeting **EURUSD** and **GBPUSD**.

---

## Architecture Overview

1. **Data & Features Engine (`src/void_fx/features/`):**
   - Stationarity via log returns.
   - Rolling 14-period Average True Range (ATR).
   - 24-hour cyclical time encodings ($\sin/\cos$).
   - Symmetrical vs. antisymmetrical feature transforms with mirror augmentation.
   - Wilder's Average Directional Index (ADX) & Rolling Hurst Exponent regime metrics.

2. **Bidirectional Cascade Alpha Engine (`src/void_fx/labeling/cascade.py` & `models/cascade/`):**
   - **Stage 1 (Direction Discovery):** XGBoost + PyTorch LSTM trained on symmetric ($\pm 1.0 \times \text{ATR}$) barriers with mirror data augmentation (balanced 50% Long / 50% Short).
   - **Stage 2 (Meta-Manager Risk Gate):** Logistic Regression evaluating the 2:1 asymmetric reward:risk bracket ($\text{TP}=+2.0\text{ATR}, \text{SL}=-1.0\text{ATR}$) in the proposed direction. Only permits trade execution if expansion probability exceeds the breakeven threshold.

3. **Execution & Risk Shield (`src/void_fx/live/cascade_runner.py`):**
   - Hourly cycle operating on strictly closed bars (`df.iloc[:-1]`).
   - Dynamic lot sizing budgeting strict 1% account equity risk.
   - Bidirectional execution:
     - **BUY:** Enters at Ask, Stop Loss below ($\text{Ask} - 1.0\text{ATR}$), Take Profit above ($\text{Ask} + 2.0\text{ATR}$).
     - **SELL:** Enters at Bid, Stop Loss above ($\text{Bid} + 1.0\text{ATR}$), Take Profit below ($\text{Bid} - 2.0\text{ATR}$).
   - Spread Shield: Automatically aborts execution if spread widens beyond 2.5 pips.

---

## Quickstart & Commands

### 1. Run Automated Test Suite
```powershell
pytest tests/test_cascade.py
```

### 2. Retrain Bidirectional Cascade Models
```powershell
python scripts/train_cascade.py
```

### 3. Run the Live Trading Robot (New Bidirectional Cascade Engine)
```powershell
python -u scripts/live_cascade_expert.py | Tee-Object -FilePath "trading_logs.txt"
```

### 4. Run Legacy Bot (Original Long-Only Prototype Fallback)
```powershell
python -u scripts/live_forex_expert.py | Tee-Object -FilePath "legacy_trading_logs.txt"
```