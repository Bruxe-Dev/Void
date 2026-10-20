# VOID TRADER & VOID VAULT — DAILY OPERATIONS & COMMANDS GUIDE

This file contains every verified command you need to test, run, monitor, evaluate, and safely stop **Void Trader** (`D:\Void\void-trader`) and **Void Vault** (`D:\Void\void-vault`), along with **what each command does** and **when to run it**.

---

# PART 1 — VOID TRADER (`D:\Void\void-trader`)

All Void Trader commands below use your dedicated Conda Python environment (`C:\Users\HP\anaconda3\envs\void\python.exe`).
Before running any Void Trader command, open **PowerShell** and move into the `void-trader` folder:

```powershell
cd D:\Void\void-trader
```

---

## 1. Switching from the Old Running Bot to the Upgraded Bot (Run Tomorrow After Your Current Trade Closes)

### A. Check if an old bot process is still running
```powershell
Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Select-Object ProcessId, CommandLine | Format-List
```
- **What it does:** Lists all active Python processes and shows the exact script they are running.
- **When to run it:** Tomorrow before starting the upgraded bot, so you don't accidentally run two copies of the bot at the same time.

### B. Stop the old bot process cleanly
- **Option 1 (Preferred):** Click into the PowerShell terminal window where the bot is running and press **`Ctrl + C`**.
- **Option 2 (By Process ID):** If you want to stop it from another PowerShell window using the `ProcessId` shown above:
  ```powershell
  Stop-Process -Id <ProcessId>
  ```
- **What it does:** Stops the Python loop without closing MetaTrader 5 or touching any open positions on the broker server.
- **When to run it:** Once your current open trade has closed (or whenever you are ready to restart with the upgraded risk gate).

---

## 2. Testing & Pre-Flight Verification Commands

### A. Run the Full Automated Test Suite (20 Tests)
```powershell
C:\Users\HP\anaconda3\envs\void\python.exe -m pytest tests -v -p no:cacheprovider
```
- **What it does:** Runs all 20 unit, integration, regression, backtester, and model-loading tests (`tests/test_cascade.py` and `tests/test_risk_and_backtest.py`), including deterministic checks for `$50`, `$100`, `$500`, and `$100,000` accounts. Never sends orders.
- **When to run it:**
  - Tomorrow morning before launching the live demo bot.
  - Anytime you edit configuration (`src/void_fx/config.py`) or code.

### B. Run a Single Live Dry-Run Cycle Against MT5 (Zero Orders Sent)
```powershell
C:\Users\HP\anaconda3\envs\void\python.exe -u scripts/live_cascade_expert.py --dry-run --once
```
- **What it does:** Connects to your running MT5 terminal, downloads the latest 500 H1 bars, computes closed-bar features, runs the XGBoost + LSTM + Meta-Gate models, and passes any trade signal through the new `OrderGate` and `mt5.order_check` **without actually sending an order** (`--dry-run`), then exits immediately (`--once`).
- **When to run it:** Right before you start the continuous live demo bot tomorrow, to confirm MT5 is connected and model inference + risk checks are healthy.

---

## 3. Starting the Live Demo Bot

### A. Start the Upgraded Bidirectional Cascade Bot (Recommended Primary Bot)
```powershell
C:\Users\HP\anaconda3\envs\void\python.exe -u scripts/live_cascade_expert.py | Tee-Object -FilePath "trading_logs.txt" -Append
```
- **What it does:** Starts the continuous H1 Bidirectional Cascade trading bot (BUY & SELL) protected by `OrderGate` (1% equity risk sizing with floor rounding, skip-on-excessive-minimum-lot, 2.5-pip spread shield, 3% daily loss limit, 10% max drawdown halt, demo-only enforcement, and bar-close cycle alignment), while displaying output in the terminal and appending to `trading_logs.txt`.
- **When to run it:** Tomorrow when you are ready for normal hourly demo trading. Leave this terminal window open.

### B. Start in Continuous Dry-Run / Paper Mode (Logs Decisions Every Hour Without Sending Orders)
```powershell
C:\Users\HP\anaconda3\envs\void\python.exe -u scripts/live_cascade_expert.py --dry-run | Tee-Object -FilePath "dry_run_logs.txt" -Append
```
- **What it does:** Runs the hourly live loop indefinitely and logs every signal and `OrderGate` sizing decision, but never calls `mt5.order_send`.
- **When to run it:** When you want to observe live signals and risk-gate decisions over several hours/days without placing demo trades.

### C. Run the Legacy v1 Bot (Long-Only Fallback — Optional)
```powershell
C:\Users\HP\anaconda3\envs\void\python.exe -u scripts/live_forex_expert.py --dry-run --once
```
- **What it does:** Runs one dry-run cycle of the original v1 3-class long-only model (now upgraded with closed-bar feature slicing and `OrderGate`). Remove `--dry-run --once` only if you specifically want to run the old v1 model instead of the v2 Cascade model.
- **When to run it:** Only for legacy comparison; do **not** run `live_forex_expert.py` and `live_cascade_expert.py` at the same time on the same account.

---

## 4. Emergency Controls, Monitoring & Resetting Risk State

### A. Trigger an Instant Emergency Stop (Blocks New Orders Without Stopping the Process)
```powershell
New-Item -ItemType File -Path D:\Void\void-trader\STOP_TRADING -Force
```
- **What it does:** Creates the sentinel file `D:\Void\void-trader\STOP_TRADING`. `OrderGate` checks for this file immediately before submitting any order; while it exists, all new orders are rejected (`REJECTED: emergency stop is active`), while open positions keep their server-side SL/TP.
- **When to run it:** During major market events (e.g., FOMC/NFP) or anytime you want to pause new entries immediately without killing the running terminal.

### B. Resume Trading After an Emergency Stop
```powershell
Remove-Item -Path D:\Void\void-trader\STOP_TRADING -ErrorAction SilentlyContinue
```
- **What it does:** Removes the `STOP_TRADING` file so `OrderGate` allows new orders again.
- **When to run it:** When you are ready to resume normal demo trading.

### C. Inspect Persistent Risk State (`state/risk_state.json`)
```powershell
Get-Content D:\Void\void-trader\state\risk_state.json
```
- **What it does:** Displays the current UTC risk day, `day_start_equity` (used for the 3% daily loss limit), `peak_equity` (used for the 10% max drawdown halt), `halted` status, and the last submitted bar timestamp per symbol (used for duplicate-order prevention).
- **When to run it:** Anytime you want to check why the risk gate halted trading or what equity high-water mark it is tracking.

### D. Reset a Max-Drawdown Halt or Reset Equity Watermarks (e.g., After Switching Demo Accounts)
```powershell
Remove-Item -Path D:\Void\void-trader\state\risk_state.json -ErrorAction SilentlyContinue
```
- **What it does:** Deletes `state/risk_state.json` so that on the next cycle `OrderGate` initializes a fresh `day_start_equity` and `peak_equity` from your current MT5 account equity.
- **When to run it:**
  - **Crucial:** Run this whenever you switch MT5 from a `$100,000` demo account to a smaller demo account (e.g., `$50` or `$500`), so the bot does not compare your new `$500` equity against the old `$100,000` `peak_equity` and trigger a drawdown halt!

---

## 5. Backtesting, Evaluation & Candidate Retraining Commands

### A. Run the Full H1 Baseline Evaluation, Account-Size Backtests ($50, $100, $500, $100k) & Experiments
```powershell
C:\Users\HP\anaconda3\envs\void\python.exe scripts/evaluate_h1_baseline.py
```
- **What it does:** Evaluates the saved H1 Cascade and Legacy models out-of-sample on `EURUSD` and `GBPUSD`, prints predictive accuracy/AUC, runs account-aware backtests for `$50`, `$100`, `$500`, and `$100,000` starting balances (with rejection breakdowns), and compares 5 candidate improvements against the unchanged baseline.
- **When to run it:** Whenever you want to review historical performance, test how different account sizes behave, or compare strategy experiments offline (does not require MT5 to be open).

### B. Train a Separate Candidate Model (Without Overwriting Your Working Baseline)
```powershell
C:\Users\HP\anaconda3\envs\void\python.exe scripts/train_cascade.py --output-dir models/cascade_candidate
```
- **What it does:** Trains a fresh Stage-1 (XGBoost + LSTM) and Stage-2 (Meta-Gate) ensemble using per-symbol 3D sequences and saves the artifacts to `models/cascade_candidate/`, leaving your active `models/cascade/` checkpoints untouched.
- **When to run it:** Only when you download new historical bars or want to test a newly trained candidate model before promoting it.

### C. Calibrate Regime Filter Thresholds (Read-Only Research Script)
```powershell
C:\Users\HP\anaconda3\envs\void\python.exe scripts/calibrate_regime.py
```
- **What it does:** Sweeps ADX enter/exit and Hurst thresholds on the training split and writes `regime_calibration.json`.
- **When to run it:** Only during offline research if you decide to recalibrate the ADX/Hurst regime filter.

---

# PART 2 — VOID VAULT (`D:\Void\void-vault`)

Before running any Void Vault command, open **PowerShell** and move into the `void-vault` folder:

```powershell
cd D:\Void\void-vault
```

### A. Build Release Binaries & Run All 5 CTest Suites (One-Command Script)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -BuildType Release -BuildDir build
```
- **What it does:** Puts MSYS2 UCRT64 (`C:\msys64\ucrt64\bin`) at the front of `PATH` (preventing Anaconda DLL conflicts), configures CMake in `build/`, compiles `void_vault.exe` and all test executables with static runtime linking, and runs all 5 test suites (`storage_test`, `auth_test`, `encryption_test`, `vault_test`, `app_e2e_test`).
- **When to run it:** After any C++ code edit or to verify the build and test suite.

### B. Build Debug Binaries & Run All 5 CTest Suites
```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -BuildType Debug -BuildDir build-debug
```
- **What it does:** Builds and tests the unoptimized Debug configuration in `build-debug/`.
- **When to run it:** When debugging C++ changes with `gdb`.

### C. Run the Interactive Void Vault Application
```powershell
.\build\void_vault.exe --data-dir .
```
- **What it does:** Launches the interactive encrypted local vault CLI using `vault_auth.dat` and `vault_data.dat` in the specified `--data-dir`. Disables console echo when reading passwords and locks/wipes session memory on exit (`6` or `quit`).
- **When to run it:** When you want to initialize an account, unlock the local vault, save encrypted text, or load stored confidential text.
