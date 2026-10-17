"""
Legacy live runner (v1 models: long-only XGBoost + LSTM + Logistic Regression stack).

This is the original live_forex_expert.py, refactored into the package with
IDENTICAL trading behaviour. Helpers now live in:
    void_fx.models.lstm        -> ForexLSTMAnalyst
    void_fx.features.base      -> prepare_live_features
    void_fx.features.regime    -> evaluate_live_regime (shadow mode by default)
    void_fx.execution.risk     -> calculate_position_size, get_compatible_filling_mode
    void_fx.data.mt5_client    -> connect_to_mt5, rates_to_frame

Entry point: scripts/live_forex_expert.py
"""

import pickle
import time
from datetime import datetime

import MetaTrader5 as mt5
import numpy as np
import torch

from void_fx.config import (
    BARS_TO_FETCH, CYCLE_INTERVAL_SECONDS, LEGACY_MODELS_DIR, MAGIC_NUMBER,
    REGIME_VETO_ENABLED, SYMBOL, ACCOUNT_NUMBER,
)
from void_fx.data.mt5_client import connect_to_mt5, rates_to_frame
from void_fx.execution.risk import calculate_position_size, get_compatible_filling_mode
from void_fx.features.base import prepare_live_features
from void_fx.features.regime import evaluate_live_regime
from void_fx.models.lstm import ForexLSTMAnalyst

TIMEFRAME = mt5.TIMEFRAME_H1

# Global model instances
xgb_analyst = None
lstm_analyst = None
meta_manager = None


def load_models():
    """Loads and compiles all 3 layers of the AI Boardroom."""
    global xgb_analyst, lstm_analyst, meta_manager
    print("[*] Loading serialized models from local memory cache...")

    # 1. XGBoost
    with open(LEGACY_MODELS_DIR / "xgb_analyst.pkl", "rb") as f:
        xgb_analyst = pickle.load(f)
    print(" [+] XGBoost Analyst loaded.")

    # 2. PyTorch LSTM
    lstm_analyst = ForexLSTMAnalyst(input_dim=4, hidden_dim=64, num_layers=2, output_dim=3)
    lstm_analyst.load_state_dict(torch.load(LEGACY_MODELS_DIR / "lstm_analyst.pth", weights_only=True))
    lstm_analyst.eval()
    print(" [+] PyTorch LSTM Analyst loaded.")

    # 3. Meta-Manager (Logistic Regression)
    with open(LEGACY_MODELS_DIR / "meta_manager.pkl", "rb") as f:
        meta_manager = pickle.load(f)
    print(" [+] Meta-Manager Stack loaded.")


def run_trading_cycle():
    global xgb_analyst, lstm_analyst, meta_manager
    if xgb_analyst is None or lstm_analyst is None or meta_manager is None:
        load_models()

    print(f"\n{'='*60}")
    print(f"[*] Execution cycle triggered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")

    if not connect_to_mt5(SYMBOL):
        print("[!] Execution cycle aborted due to connection failure.")
        return

    # Fetch historical hourly bars
    raw_rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS_TO_FETCH)
    account_info = mt5.account_info()

    if raw_rates is None or len(raw_rates) == 0:
        print(f"[!] Failed to fetch market rates for {SYMBOL}: {mt5.last_error()}")
        mt5.shutdown()
        return

    df = rates_to_frame(raw_rates)

    try:
        X_2d_live, X_3d_live, current_atr = prepare_live_features(df)
    except Exception as e:
        print(f"[!] Feature engineering failed: {e}")
        mt5.shutdown()
        return

    # Layer 1: XGBoost tabular inference
    xgb_probs = xgb_analyst.predict_proba(X_2d_live)

    # Layer 2: PyTorch LSTM sequential inference
    X_3d_tensor = torch.tensor(X_3d_live, dtype=torch.float32)
    with torch.no_grad():
        lstm_logits = lstm_analyst(X_3d_tensor)
        lstm_probs = torch.nn.Softmax(dim=1)(lstm_logits).numpy()

    # Layer 3: Blended feature stack into Logistic Regression Meta-Manager
    X_blend_live = np.hstack((xgb_probs, lstm_probs))
    final_signal = meta_manager.predict(X_blend_live)[0]
    meta_probs = meta_manager.predict_proba(X_blend_live)[0]

    # Telemetry logging
    signal_labels = {0: "Stop Loss / Bearish", 1: "Time Expiry / Neutral", 2: "Take Profit / Bullish"}
    print(f"[*] Market Telemetry:")
    print(f"    - Current ATR(14): {current_atr:.5f}")
    print(f"    - XGBoost Probabilities [SL, Expiry, TP]: {np.round(xgb_probs[0], 4).tolist()}")
    print(f"    - LSTM Probabilities    [SL, Expiry, TP]: {np.round(lstm_probs[0], 4).tolist()}")
    print(f"    - Meta-Manager Probs    [SL, Expiry, TP]: {np.round(meta_probs, 4).tolist()}")
    print(f"    - Selected Model Intent: Class {final_signal} ({signal_labels.get(final_signal, 'Unknown')})")

    # ------------------------------------------------------------------
    # STEP 1: REGIME FILTER (hard override of the Meta-Manager)
    # Evaluated on CLOSED bars only: copy_rates_from_pos(..., 0, N) puts the
    # still-forming candle in the last row, so we drop it. The gate can only
    # veto trades (force Class 1); it can never create one. Any internal error
    # inside evaluate_live_regime() resolves to regime_on=False (fail-safe).
    # ------------------------------------------------------------------
    regime = evaluate_live_regime(df.iloc[:-1])
    print(f"    - Regime Filter [{regime.bar_time}]: ADX {regime.adx:.2f} "
          f"(+DI {regime.plus_di:.2f} / -DI {regime.minus_di:.2f}) | Hurst {regime.hurst:.3f} "
          f"-> {'ON' if regime.regime_on else 'OFF'} ({regime.reason})")
    if not regime.regime_on and final_signal != 1:
        if REGIME_VETO_ENABLED:
            print(f"    - [REGIME VETO] Meta-Manager intent Class {final_signal} overridden -> Class 1 (Stand Down).")
            final_signal = 1
        else:
            print(f"    - [REGIME SHADOW] Filter WOULD VETO Class {final_signal} (veto disabled, signal unchanged).")

    # Check for existing positions to prevent duplicate entries
    open_positions = mt5.positions_get(symbol=SYMBOL)
    bot_positions = [p for p in open_positions if p.magic == MAGIC_NUMBER] if open_positions else []

    if final_signal == 2:
        if len(bot_positions) > 0:
            print(f"[!] BUY Signal detected, but {len(bot_positions)} active trade(s) already running (Ticket: {bot_positions[0].ticket}). Skipping new order to respect risk limits.")
        else:
            live_balance = account_info.balance
            calculated_lots = calculate_position_size(live_balance, current_atr, SYMBOL)
            symbol_info = mt5.symbol_info(SYMBOL)
            tick = mt5.symbol_info_tick(SYMBOL)

            if symbol_info is None or tick is None:
                print(f"[!] Unable to retrieve current live tick for {SYMBOL}.")
                mt5.shutdown()
                return

            filling_mode = get_compatible_filling_mode(symbol_info)
            digits = symbol_info.digits
            ask_price = round(tick.ask, digits)
            sl_price = round(ask_price - current_atr, digits)
            tp_price = round(ask_price + (current_atr * 2.0), digits)

            print(f"[+] BUY SIGNAL VALIDATED!")
            print(f"    - Account Balance: ${live_balance:,.2f}")
            print(f"    - Calculated Risk Volume: {calculated_lots} Lots")
            print(f"    - Order Ask Price: {ask_price} | SL: {sl_price} | TP: {tp_price}")

            order_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": calculated_lots,
                "type": mt5.ORDER_TYPE_BUY,
                "price": ask_price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 10,
                "magic": MAGIC_NUMBER,
                "comment": "AI TExpert Active Trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }

            # Pre-flight order check
            check_result = mt5.order_check(order_request)
            if check_result is None or check_result.retcode != 0:
                print(f"[!] Pre-order check failed: {check_result.comment if check_result else mt5.last_error()}")
            else:
                result = mt5.order_send(order_request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"[+] Order successfully executed! Deal Ticket: {result.deal}, Order Ticket: {result.order}")
                else:
                    retcode_msg = result.comment if result else mt5.last_error()
                    print(f"[!] Order rejected by server. Code: {result.retcode if result else 'FAILED'}, Comment: {retcode_msg}")
    else:
        print(f"[-] System evaluates market state as flat/unsafe (Class {final_signal}). Standing down.")

    mt5.shutdown()
    print(f"[*] Cycle finished. Next execution in {CYCLE_INTERVAL_SECONDS // 60} minutes.")
    print(f"{'='*60}\n")


def main():
    print("=" * 60)
    print("AI BOARDROOM FOREX EXPERT ADVISOR - INITIALIZATION")
    print("=" * 60)
    load_models()

    print("\n[*] Initializing Live Trading Engine...")
    print(f"[*] Target Symbol: {SYMBOL} | Timeframe: H1 | Account: {ACCOUNT_NUMBER}")
    print("[*] Press Ctrl+C at any time to safely exit.")

    try:
        while True:
            run_trading_cycle()
            time.sleep(CYCLE_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n[!] KeyboardInterrupt received. Gracefully shutting down MT5 and exiting...")
        mt5.shutdown()
        print("[*] Bot successfully stopped.")