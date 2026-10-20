"""
Bidirectional Cascade Live Runner.
Features:
  - Stage 1: Predicts Direction (Buy vs Sell) using XGBoost + LSTM on symmetric barriers.
  - Stage 2: Meta-Gate evaluates risk/reward expansion probability.
  - Executes BUY (at Ask, SL below, TP above) AND SELL (at Bid, SL above, TP below).
  - Handles spread widening, slippage deviation, filling modes, and account equity risk.

Prediction logic is unchanged from the original runner (now in predict_cascade(), which
the backtester also uses). Execution changes (defect fixes):
  - every order goes through execution.order_gate.OrderGate (account-size-aware sizing
    from live equity, hard risk limits, demo-only guard, duplicate protection);
  - a failing cycle is logged and the loop continues (previously any exception, e.g.
    account_info() returning None, killed the bot);
  - positions_get() returning None (terminal error) aborts the cycle instead of being
    treated as "no open positions" (which could duplicate orders);
  - non-finite features abort the cycle;
  - cycles are aligned to H1 bar close (config.CYCLE_ALIGN_TO_BAR_CLOSE).

Usage: scripts/live_cascade_expert.py [--dry-run] [--once]
"""

import argparse
import json
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
    CASCADE_MODELS_DIR,
    CYCLE_ALIGN_DELAY_SECONDS,
    CYCLE_ALIGN_TO_BAR_CLOSE,
    CYCLE_INTERVAL_SECONDS,
    LOOKBACK_WINDOW,
    MAGIC_NUMBER,
    PT_MULT,
    SL_MULT,
    SYMBOL,
)
from void_fx.data.mt5_client import connect_to_mt5, rates_to_frame
from void_fx.execution.order_gate import OrderGate
from void_fx.execution.risk_manager import BUY, SELL, OrderIntent
from void_fx.features.cascade import (
    CASCADE_FEATURE_COLS,
    FeatureScaler,
    build_cascade_features,
)
from void_fx.models.lstm import ForexLSTMAnalyst

# Execution hyperparameters
TIMEFRAME = mt5.TIMEFRAME_H1
MAX_SPREAD_PIPS = 2.5       # Spread shield: skip execution if broker spread widens past 2.5 pips
PROBABILITY_THRESHOLD = 0.50 # Meta-Gate entry cutoff

# Model cache
xgb_dir = None
lstm_dir = None
meta_gate = None
scaler = None


def load_cascade_models():
    global xgb_dir, lstm_dir, meta_gate, scaler
    print("[*] Loading Bidirectional Cascade Models from disk...")

    with open(CASCADE_MODELS_DIR / "scaler.json", "r") as f:
        scaler = FeatureScaler.from_dict(json.load(f))

    with open(CASCADE_MODELS_DIR / "xgb_dir.pkl", "rb") as f:
        xgb_dir = pickle.load(f)

    with open(CASCADE_MODELS_DIR / "meta_gate.pkl", "rb") as f:
        meta_gate = pickle.load(f)

    lstm_dir = ForexLSTMAnalyst(
        input_dim=len(CASCADE_FEATURE_COLS),
        hidden_dim=64,
        output_dim=2,
        num_layers=2
    )
    lstm_dir.load_state_dict(torch.load(CASCADE_MODELS_DIR / "lstm_dir.pth", weights_only=True))
    lstm_dir.eval()
    print("[+] All Bidirectional Cascade Models loaded successfully!")


def predict_cascade(df_closed):
    """
    Pure inference on CLOSED bars (identical math to the original live cycle).
    Returns dict(atr, xgb, lstm, p_long, p_short, direction, p_expansion, bar_time).
    Raises ValueError on insufficient or non-finite inputs.
    """
    if xgb_dir is None or lstm_dir is None or meta_gate is None or scaler is None:
        load_cascade_models()
    feats_df = build_cascade_features(df_closed)
    valid_feats = feats_df.dropna(subset=CASCADE_FEATURE_COLS)
    if len(valid_feats) < LOOKBACK_WINDOW:
        raise ValueError(f"Insufficient history: {len(valid_feats)} < {LOOKBACK_WINDOW}")

    current_atr = valid_feats["volatility_atr"].iloc[-1]
    X_raw = valid_feats[CASCADE_FEATURE_COLS].values
    X_norm = scaler.transform(X_raw)
    X_2d_live = X_norm[-1:]
    X_3d_live = np.array([X_norm[-LOOKBACK_WINDOW:]], dtype=np.float32)
    if not (np.isfinite(X_3d_live).all() and np.isfinite(current_atr) and current_atr > 0):
        raise ValueError("Non-finite features or ATR in the inference window")
    if valid_feats.index[-1] != df_closed.index[-1]:
        raise ValueError(f"Latest closed bar {df_closed.index[-1]} has no valid features")

    # STAGE 1: DIRECTION INFERENCE
    xgb_dir_probs = xgb_dir.predict_proba(X_2d_live)[0]  # [P(Short), P(Long)]
    with torch.no_grad():
        lstm_logits = lstm_dir(torch.tensor(X_3d_live))
        lstm_dir_probs = torch.softmax(lstm_logits, dim=1).numpy()[0]

    # Blended direction score
    p_long_blend = 0.5 * xgb_dir_probs[1] + 0.5 * lstm_dir_probs[1]
    p_short_blend = 0.5 * xgb_dir_probs[0] + 0.5 * lstm_dir_probs[0]
    proposed_direction = 1 if p_long_blend >= p_short_blend else -1

    # STAGE 2: META-MANAGER GATE INFERENCE
    X_meta_live = np.hstack([X_2d_live, [xgb_dir_probs]])
    p_expansion = meta_gate.predict_proba(X_meta_live)[0, 1]
    return dict(atr=float(current_atr), xgb=xgb_dir_probs, lstm=lstm_dir_probs, p_long=float(p_long_blend),
                p_short=float(p_short_blend), direction=proposed_direction, p_expansion=float(p_expansion),
                bar_time=df_closed.index[-1])


def run_cascade_trading_cycle(gate: OrderGate):
    print(f"\n{'='*60}")
    print(f"[*] Bidirectional cycle triggered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
          f"{' [DRY RUN]' if gate.dry_run else ''}...")

    if not connect_to_mt5(SYMBOL):
        print("[!] Execution cycle aborted due to MT5 connection failure.")
        return

    try:
        raw_rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS_TO_FETCH)
        if raw_rates is None or len(raw_rates) == 0:
            print(f"[!] Failed to fetch market rates for {SYMBOL}: {mt5.last_error()}")
            return

        df_raw = rates_to_frame(raw_rates)
        # Exclude currently forming bar to prevent lookahead leak
        df_closed = df_raw.iloc[:-1].copy()

        try:
            pred = predict_cascade(df_closed)
        except Exception as e:
            print(f"[!] Feature preparation failed: {e}")
            return

        dir_label = "BUY (Long)" if pred["direction"] == 1 else "SELL (Short)"
        print("[*] Market Telemetry:")
        print(f"    - Signal bar (closed): {pred['bar_time']}")
        print(f"    - Current ATR(14): {pred['atr']:.5f}")
        print(f"    - Direction Probabilities (XGBoost) [Short, Long]: {np.round(pred['xgb'], 4).tolist()}")
        print(f"    - Direction Probabilities (LSTM)    [Short, Long]: {np.round(pred['lstm'], 4).tolist()}")
        print(f"    - Blended Direction Consensus: {dir_label} (P_Long: {pred['p_long']:.4f}, P_Short: {pred['p_short']:.4f})")
        print(f"    - Meta-Gate Expansion Probability: {pred['p_expansion']:.4f} (Threshold: {PROBABILITY_THRESHOLD:.2f})")

        # Risk Shield: Active position check (None = terminal error, NOT "no positions")
        open_positions = mt5.positions_get(symbol=SYMBOL)
        if open_positions is None:
            print(f"[!] positions_get failed ({mt5.last_error()}); aborting cycle to avoid duplicate orders.")
            return
        bot_positions = [p for p in open_positions if p.magic == MAGIC_NUMBER]
        if bot_positions:
            print(f"[!] Active trade already running (Ticket: {bot_positions[0].ticket}, Type: {'BUY' if bot_positions[0].type==0 else 'SELL'}). Skipping.")
            return

        if pred["p_expansion"] < PROBABILITY_THRESHOLD:
            print(f"[-] Meta-Gate classified state as trap / low edge (P_Expansion {pred['p_expansion']:.4f} < {PROBABILITY_THRESHOLD:.2f}). Standing down.")
            return

        symbol_info = mt5.symbol_info(SYMBOL)
        tick = mt5.symbol_info_tick(SYMBOL)
        if symbol_info is None or tick is None:
            print(f"[!] Failed to fetch live tick for {SYMBOL}.")
            return

        point = symbol_info.point
        spread_pips = (tick.ask - tick.bid) / (10 * point) if point > 0 else float("inf")
        if spread_pips > MAX_SPREAD_PIPS:
            print(f"[!] SPREAD SHIELD TRIGGERED: Current spread {spread_pips:.1f} pips > {MAX_SPREAD_PIPS} pips. Execution aborted.")
            return

        digits = symbol_info.digits
        current_atr = pred["atr"]
        if pred["direction"] == 1:
            # BUY ORDER: Enter at Ask, SL below, TP above
            side, entry_price = BUY, round(tick.ask, digits)
            sl_price = round(entry_price - (current_atr * SL_MULT), digits)
            tp_price = round(entry_price + (current_atr * PT_MULT), digits)
        else:
            # SELL ORDER: Enter at Bid, SL above, TP below
            side, entry_price = SELL, round(tick.bid, digits)
            sl_price = round(entry_price + (current_atr * SL_MULT), digits)
            tp_price = round(entry_price - (current_atr * PT_MULT), digits)

        print(f"[+] TRADE SIGNAL VALIDATED! Submitting {dir_label} to the risk gate...")
        print(f"    - Entry: {entry_price} | SL: {sl_price} | TP: {tp_price}")
        intent = OrderIntent(SYMBOL, side, entry_price, sl_price, tp_price,
                             signal_bar_time=str(pred["bar_time"]), comment=f"Void Cascade {dir_label[:4]}")
        gate.submit(intent)
    finally:
        mt5.shutdown()
        print(f"[*] Cycle finished.")
        print(f"{'='*60}\n")


def seconds_until_next_cycle(now: float | None = None) -> float:
    now = time.time() if now is None else now
    if not CYCLE_ALIGN_TO_BAR_CLOSE:
        return CYCLE_INTERVAL_SECONDS
    next_hour = (int(now) // 3600 + 1) * 3600
    return max(1.0, next_hour + CYCLE_ALIGN_DELAY_SECONDS - now)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Void FX bidirectional cascade bot (demo only by default)")
    parser.add_argument("--dry-run", action="store_true", help="validate orders with the risk gate but never send them")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    args = parser.parse_args(argv)

    print("=" * 60)
    print("VOID FX: BIDIRECTIONAL CASCADE EXPERT ADVISOR - INITIALIZATION")
    print("=" * 60)
    load_cascade_models()
    print(f"[*] Target Symbol: {SYMBOL} | Timeframe: H1 | Account: {ACCOUNT_NUMBER}")
    print("[*] Press Ctrl+C to safely exit (open positions keep their broker-side SL/TP).\n")
    gate = OrderGate(magic=MAGIC_NUMBER, dry_run=args.dry_run)
    try:
        while True:
            try:
                run_cascade_trading_cycle(gate)
            except Exception:
                print("[!] Unhandled error in trading cycle (loop continues):")
                traceback.print_exc()
            if args.once:
                break
            wait = seconds_until_next_cycle()
            print(f"[*] Next execution in {wait / 60:.1f} minutes.")
            time.sleep(wait)
    except KeyboardInterrupt:
        print("\n[!] Gracefully shutting down...")
    finally:
        mt5.shutdown()
        print("[*] Bot successfully stopped.")


if __name__ == "__main__":
    main()
