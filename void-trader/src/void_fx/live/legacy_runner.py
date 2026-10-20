"""
Legacy live runner (v1 models: long-only XGBoost + LSTM + Logistic Regression stack).

This is the original live_forex_expert.py, refactored into the package with
preserved model inference. Helpers live in:
    void_fx.models.lstm        -> ForexLSTMAnalyst
    void_fx.features.base      -> prepare_live_features
    void_fx.features.regime    -> evaluate_live_regime (shadow mode by default)
    void_fx.execution.order_gate -> OrderGate
    void_fx.data.mt5_client    -> connect_to_mt5, rates_to_frame

Defect fixes applied:
  * Feature preparation now runs strictly on closed bars (df.iloc[:-1]) so the
    still-forming candle is not passed into the model.
  * Order submission routes through OrderGate (account-size-aware position sizing,
    hard risk limits, demo-only guard, and duplicate protection).
  * positions_get() returning None aborts the cycle instead of treating terminal
    failure as zero open positions.
  * Unhandled exceptions in a cycle are logged without crashing the main loop.

Entry point: scripts/live_forex_expert.py
"""

import argparse
import pickle
import time
import traceback
from datetime import datetime

import MetaTrader5 as mt5
import numpy as np
import torch

from void_fx.config import (
    ACCOUNT_NUMBER,
    BARS_TO_FETCH,
    CYCLE_INTERVAL_SECONDS,
    LEGACY_MODELS_DIR,
    MAGIC_NUMBER,
    REGIME_VETO_ENABLED,
    SYMBOL,
)
from void_fx.data.mt5_client import connect_to_mt5, rates_to_frame
from void_fx.execution.order_gate import OrderGate
from void_fx.execution.risk_manager import BUY, OrderIntent
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


def predict_legacy(df_closed):
    """
    Pure inference on CLOSED bars for the legacy v1 3-class model.
    Returns dict(atr, xgb_probs, lstm_probs, meta_probs, signal, bar_time).
    """
    if xgb_analyst is None or lstm_analyst is None or meta_manager is None:
        load_models()
    X_2d_live, X_3d_live, current_atr = prepare_live_features(df_closed)
    if not (np.isfinite(X_3d_live).all() and np.isfinite(current_atr) and current_atr > 0):
        raise ValueError("Non-finite features or ATR in legacy inference window")

    xgb_probs = xgb_analyst.predict_proba(X_2d_live)
    X_3d_tensor = torch.tensor(X_3d_live, dtype=torch.float32)
    with torch.no_grad():
        lstm_logits = lstm_analyst(X_3d_tensor)
        lstm_probs = torch.nn.Softmax(dim=1)(lstm_logits).numpy()

    X_blend_live = np.hstack((xgb_probs, lstm_probs))
    final_signal = int(meta_manager.predict(X_blend_live)[0])
    meta_probs = meta_manager.predict_proba(X_blend_live)[0]
    return dict(
        atr=float(current_atr),
        xgb_probs=xgb_probs[0],
        lstm_probs=lstm_probs[0],
        meta_probs=meta_probs,
        signal=final_signal,
        bar_time=df_closed.index[-1],
    )


def run_trading_cycle(gate: OrderGate | None = None):
    if gate is None:
        gate = OrderGate(magic=MAGIC_NUMBER)
    if xgb_analyst is None or lstm_analyst is None or meta_manager is None:
        load_models()

    print(f"\n{'='*60}")
    print(f"[*] Execution cycle triggered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")

    if not connect_to_mt5(SYMBOL):
        print("[!] Execution cycle aborted due to connection failure.")
        return

    try:
        # Fetch historical hourly bars
        raw_rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS_TO_FETCH)
        if raw_rates is None or len(raw_rates) == 0:
            print(f"[!] Failed to fetch market rates for {SYMBOL}: {mt5.last_error()}")
            return

        df = rates_to_frame(raw_rates)
        df_closed = df.iloc[:-1].copy()

        try:
            pred = predict_legacy(df_closed)
        except Exception as e:
            print(f"[!] Feature engineering failed: {e}")
            return

        current_atr = pred["atr"]
        final_signal = pred["signal"]

        # Telemetry logging
        signal_labels = {0: "Stop Loss / Bearish", 1: "Time Expiry / Neutral", 2: "Take Profit / Bullish"}
        print("[*] Market Telemetry:")
        print(f"    - Signal bar (closed): {pred['bar_time']}")
        print(f"    - Current ATR(14): {current_atr:.5f}")
        print(f"    - XGBoost Probabilities [SL, Expiry, TP]: {np.round(pred['xgb_probs'], 4).tolist()}")
        print(f"    - LSTM Probabilities    [SL, Expiry, TP]: {np.round(pred['lstm_probs'], 4).tolist()}")
        print(f"    - Meta-Manager Probs    [SL, Expiry, TP]: {np.round(pred['meta_probs'], 4).tolist()}")
        print(f"    - Selected Model Intent: Class {final_signal} ({signal_labels.get(final_signal, 'Unknown')})")

        # STEP 1: REGIME FILTER (evaluated on CLOSED bars only)
        regime = evaluate_live_regime(df_closed)
        print(f"    - Regime Filter [{regime.bar_time}]: ADX {regime.adx:.2f} "
              f"(+DI {regime.plus_di:.2f} / -DI {regime.minus_di:.2f}) | Hurst {regime.hurst:.3f} "
              f"-> {'ON' if regime.regime_on else 'OFF'} ({regime.reason})")
        if not regime.regime_on and final_signal != 1:
            if REGIME_VETO_ENABLED:
                print(f"    - [REGIME VETO] Meta-Manager intent Class {final_signal} overridden -> Class 1 (Stand Down).")
                final_signal = 1
            else:
                print(f"    - [REGIME SHADOW] Filter WOULD VETO Class {final_signal} (veto disabled, signal unchanged).")

        # Check for existing positions to prevent duplicate entries (None = terminal error)
        open_positions = mt5.positions_get(symbol=SYMBOL)
        if open_positions is None:
            print(f"[!] positions_get failed ({mt5.last_error()}); aborting cycle to avoid duplicate orders.")
            return
        bot_positions = [p for p in open_positions if p.magic == MAGIC_NUMBER]

        if final_signal == 2:
            if len(bot_positions) > 0:
                print(f"[!] BUY Signal detected, but {len(bot_positions)} active trade(s) already running "
                      f"(Ticket: {bot_positions[0].ticket}). Skipping new order to respect risk limits.")
            else:
                symbol_info = mt5.symbol_info(SYMBOL)
                tick = mt5.symbol_info_tick(SYMBOL)

                if symbol_info is None or tick is None:
                    print(f"[!] Unable to retrieve current live tick for {SYMBOL}.")
                    return

                digits = symbol_info.digits
                ask_price = round(tick.ask, digits)
                sl_price = round(ask_price - current_atr, digits)
                tp_price = round(ask_price + (current_atr * 2.0), digits)

                print("[+] BUY SIGNAL VALIDATED! Submitting to the risk gate...")
                print(f"    - Order Ask Price: {ask_price} | SL: {sl_price} | TP: {tp_price}")
                intent = OrderIntent(
                    SYMBOL,
                    BUY,
                    ask_price,
                    sl_price,
                    tp_price,
                    signal_bar_time=str(pred["bar_time"]),
                    comment="AI TExpert Active Trade",
                )
                gate.submit(intent)
        else:
            print(f"[-] System evaluates market state as flat/unsafe (Class {final_signal}). Standing down.")
    finally:
        mt5.shutdown()
        print(f"[*] Cycle finished. Next execution in {CYCLE_INTERVAL_SECONDS // 60} minutes.")
        print(f"{'='*60}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Void FX legacy v1 bot (demo only by default)")
    parser.add_argument("--dry-run", action="store_true", help="validate orders with the risk gate but never send them")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    args = parser.parse_args(argv)

    print("=" * 60)
    print("AI BOARDROOM FOREX EXPERT ADVISOR - INITIALIZATION")
    print("=" * 60)
    load_models()

    print("\n[*] Initializing Live Trading Engine...")
    print(f"[*] Target Symbol: {SYMBOL} | Timeframe: H1 | Account: {ACCOUNT_NUMBER}")
    print("[*] Press Ctrl+C at any time to safely exit.")

    gate = OrderGate(magic=MAGIC_NUMBER, dry_run=args.dry_run)
    try:
        while True:
            try:
                run_trading_cycle(gate)
            except Exception:
                print("[!] Unhandled error in trading cycle (loop continues):")
                traceback.print_exc()
            if args.once:
                break
            time.sleep(CYCLE_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n[!] KeyboardInterrupt received. Gracefully shutting down MT5 and exiting...")
    finally:
        mt5.shutdown()
        print("[*] Bot successfully stopped.")