"""
Bidirectional Cascade Live Runner.
Features:
  - Stage 1: Predicts Direction (Buy vs Sell) using XGBoost + LSTM on symmetric barriers.
  - Stage 2: Meta-Gate evaluates risk/reward expansion probability.
  - Executes BUY (at Ask, SL below, TP above) AND SELL (at Bid, SL above, TP below).
  - Handles spread widening, slippage deviation, filling modes, and account equity risk.
"""

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import torch

from void_fx.config import (
    ACCOUNT_NUMBER,
    BARS_TO_FETCH,
    CASCADE_MODELS_DIR,
    CYCLE_INTERVAL_SECONDS,
    LOOKBACK_WINDOW,
    MAGIC_NUMBER,
    MT5_PATH,
    PT_MULT,
    RISK_PERCENTAGE,
    SL_MULT,
    SYMBOL,
)
from void_fx.data.mt5_client import connect_to_mt5, rates_to_frame
from void_fx.execution.risk import calculate_position_size, get_compatible_filling_mode
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


def run_cascade_trading_cycle():
    global xgb_dir, lstm_dir, meta_gate, scaler
    if xgb_dir is None or lstm_dir is None or meta_gate is None:
        load_cascade_models()

    print(f"\n{'='*60}")
    print(f"[*] Bidirectional cycle triggered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")

    if not connect_to_mt5(SYMBOL):
        print("[!] Execution cycle aborted due to MT5 connection failure.")
        return

    # Ingest historical bars
    raw_rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS_TO_FETCH)
    account_info = mt5.account_info()

    if raw_rates is None or len(raw_rates) == 0:
        print(f"[!] Failed to fetch market rates for {SYMBOL}: {mt5.last_error()}")
        mt5.shutdown()
        return

    df_raw = rates_to_frame(raw_rates)
    
    # Exclude currently forming bar to prevent lookahead leak
    df_closed = df_raw.iloc[:-1].copy()
    
    try:
        feats_df = build_cascade_features(df_closed)
        valid_feats = feats_df.dropna(subset=CASCADE_FEATURE_COLS)
        
        if len(valid_feats) < LOOKBACK_WINDOW:
            raise ValueError(f"Insufficient history: {len(valid_feats)} < {LOOKBACK_WINDOW}")
            
        current_atr = valid_feats["volatility_atr"].iloc[-1]
        X_raw = valid_feats[CASCADE_FEATURE_COLS].values
        X_norm = scaler.transform(X_raw)
        
        # 2D feature for XGBoost
        X_2d_live = X_norm[-1:]
        
        # 3D tensor for LSTM
        X_3d_live = np.array([X_norm[-LOOKBACK_WINDOW:]], dtype=np.float32)
    except Exception as e:
        print(f"[!] Feature preparation failed: {e}")
        mt5.shutdown()
        return

    # STAGE 1: DIRECTION INFERENCE
    xgb_dir_probs = xgb_dir.predict_proba(X_2d_live)[0]  # [P(Short), P(Long)]
    
    with torch.no_grad():
        lstm_logits = lstm_dir(torch.tensor(X_3d_live))
        lstm_dir_probs = torch.softmax(lstm_logits, dim=1).numpy()[0]
        
    # Blended direction score
    p_long_blend = 0.5 * xgb_dir_probs[1] + 0.5 * lstm_dir_probs[1]
    p_short_blend = 0.5 * xgb_dir_probs[0] + 0.5 * lstm_dir_probs[0]
    
    proposed_direction = 1 if p_long_blend >= p_short_blend else -1
    dir_label = "BUY (Long)" if proposed_direction == 1 else "SELL (Short)"

    # STAGE 2: META-MANAGER GATE INFERENCE
    X_meta_live = np.hstack([X_2d_live, [xgb_dir_probs]])
    p_expansion = meta_gate.predict_proba(X_meta_live)[0, 1]

    # Telemetry logging
    print("[*] Market Telemetry:")
    print(f"    - Current ATR(14): {current_atr:.5f}")
    print(f"    - Direction Probabilities (XGBoost) [Short, Long]: {np.round(xgb_dir_probs, 4).tolist()}")
    print(f"    - Direction Probabilities (LSTM)    [Short, Long]: {np.round(lstm_dir_probs, 4).tolist()}")
    print(f"    - Blended Direction Consensus: {dir_label} (P_Long: {p_long_blend:.4f}, P_Short: {p_short_blend:.4f})")
    print(f"    - Meta-Gate Expansion Probability: {p_expansion:.4f} (Threshold: {PROBABILITY_THRESHOLD:.2f})")

    # Risk Shield: Active position check
    open_positions = mt5.positions_get(symbol=SYMBOL)
    bot_positions = [p for p in open_positions if p.magic == MAGIC_NUMBER] if open_positions else []
    
    if len(bot_positions) > 0:
        print(f"[!] Active trade already running (Ticket: {bot_positions[0].ticket}, Type: {'BUY' if bot_positions[0].type==0 else 'SELL'}). Skipping.")
        mt5.shutdown()
        return

    # Evaluate execution gate
    if p_expansion >= PROBABILITY_THRESHOLD:
        symbol_info = mt5.symbol_info(SYMBOL)
        tick = mt5.symbol_info_tick(SYMBOL)

        if symbol_info is None or tick is None:
            print(f"[!] Failed to fetch live tick for {SYMBOL}.")
            mt5.shutdown()
            return

        point = symbol_info.point
        spread_pips = (tick.ask - tick.bid) / (10 * point) if point > 0 else 0
        if spread_pips > MAX_SPREAD_PIPS:
            print(f"[!] SPREAD SHIELD TRIGGERED: Current spread {spread_pips:.1f} pips > {MAX_SPREAD_PIPS} pips. Execution aborted.")
            mt5.shutdown()
            return

        live_balance = account_info.balance
        calculated_lots = calculate_position_size(live_balance, current_atr, SYMBOL, risk_pct=RISK_PERCENTAGE)
        filling_mode = get_compatible_filling_mode(symbol_info)
        digits = symbol_info.digits

        if proposed_direction == 1:
            # BUY ORDER: Enter at Ask, SL below, TP above
            order_type = mt5.ORDER_TYPE_BUY
            entry_price = round(tick.ask, digits)
            sl_price = round(entry_price - (current_atr * SL_MULT), digits)
            tp_price = round(entry_price + (current_atr * PT_MULT), digits)
        else:
            # SELL ORDER: Enter at Bid, SL above, TP below
            order_type = mt5.ORDER_TYPE_SELL
            entry_price = round(tick.bid, digits)
            sl_price = round(entry_price + (current_atr * SL_MULT), digits)
            tp_price = round(entry_price - (current_atr * PT_MULT), digits)

        print(f"[+] TRADE SIGNAL VALIDATED! Executing {dir_label}...")
        print(f"    - Lots: {calculated_lots} | Entry: {entry_price} | SL: {sl_price} | TP: {tp_price}")

        order_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": calculated_lots,
            "type": order_type,
            "price": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 10,
            "magic": MAGIC_NUMBER,
            "comment": f"Void Cascade {dir_label[:4]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        check_res = mt5.order_check(order_request)
        if check_res is None or check_res.retcode != 0:
            print(f"[!] Order pre-check rejected: {check_res.comment if check_res else mt5.last_error()}")
        else:
            result = mt5.order_send(order_request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[+] Order executed successfully! Deal: {result.deal}, Ticket: {result.order}")
            else:
                print(f"[!] Order rejected by server: {result.comment if result else mt5.last_error()}")
    else:
        print(f"[-] Meta-Gate classified state as trap / low edge (P_Expansion {p_expansion:.4f} < {PROBABILITY_THRESHOLD:.2f}). Standing down.")

    mt5.shutdown()
    print(f"[*] Cycle finished. Next execution in {CYCLE_INTERVAL_SECONDS // 60} minutes.")
    print(f"{'='*60}\n")


def main():
    print("=" * 60)
    print("VOID FX: BIDIRECTIONAL CASCADE EXPERT ADVISOR - INITIALIZATION")
    print("=" * 60)
    load_cascade_models()
    print(f"[*] Target Symbol: {SYMBOL} | Timeframe: H1 | Account: {ACCOUNT_NUMBER}")
    print("[*] Press Ctrl+C to safely exit.\n")
    try:
        while True:
            run_cascade_trading_cycle()
            time.sleep(CYCLE_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n[!] Gracefully shutting down...")
        mt5.shutdown()
        print("[*] Bot successfully stopped.")


if __name__ == "__main__":
    main()
