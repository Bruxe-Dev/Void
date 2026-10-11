import time
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
LOOKBACK_WINDOW = 24
RISK_PERCENTAGE = 0.01  
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
ACCOUNT_NUMBER = 5057177985
MAGIC_NUMBER = 202610
BARS_TO_FETCH = 100
CYCLE_INTERVAL_SECONDS = 3600

# ==========================================
# NEURAL NETWORK ARCHITECTURE
# ==========================================
class ForexLSTMAnalyst(nn.Module):
    """
    Long Short-Term Memory (LSTM) network matching the architecture 
    trained in sprint1-data-engine.ipynb.
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = 3, num_layers: int = 2):
        super(ForexLSTMAnalyst, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0 
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        final_time_step_output = out[:, -1, :]
        return self.fc(final_time_step_output)


def calculate_position_size(balance: float, atr: float, symbol: str) -> float:
    """
    Computes volatility-adjusted position size based on cash risk budget and ATR stop distance.
    """
    cash_risk_budget = balance * RISK_PERCENTAGE
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return 0.01 
        
    trade_tick_size = symbol_info.trade_tick_size
    trade_tick_value = symbol_info.trade_tick_value
    volume_step = symbol_info.volume_step
    
    # 1 lot risk for 1 ATR movement: atr / tick_size * tick_value
    risk_per_lot = (atr / trade_tick_size) * trade_tick_value
    
    if risk_per_lot <= 0:
        return symbol_info.volume_min
        
    raw_lots = cash_risk_budget / risk_per_lot
    
    # Precision rounding according to broker volume step
    step_precision = len(str(volume_step).split('.')[1]) if '.' in str(volume_step) else 0
    final_lots = round(round(raw_lots / volume_step) * volume_step, step_precision)
    return max(symbol_info.volume_min, min(symbol_info.volume_max, final_lots))


def get_compatible_filling_mode(symbol_info) -> int:
    """
    Resolves the appropriate order execution filling mode supported by the broker.
    """
    if symbol_info.filling_mode & 1:
        return mt5.ORDER_FILLING_FOK
    elif symbol_info.filling_mode & 2:
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_RETURN


def prepare_live_features(df: pd.DataFrame) -> tuple:
    """
    Transforms raw incoming OHLCV bars into aligned 2D (tabular) and 3D (temporal tensor) features.
    Maintains exact column sequence matching the training phase:
    ['log_return', 'volatility_atr', 'hour_cos', 'hour_sin']
    """
    matrix = df.copy()
    matrix['log_return'] = np.log(matrix['close'] / matrix['close'].shift(1))
    matrix['high_low_span'] = matrix['high'] - matrix['low']
    matrix['volatility_atr'] = matrix['high_low_span'].rolling(window=14).mean()
    matrix['hour'] = matrix.index.hour
    matrix['hour_sin'] = np.sin(2 * np.pi * matrix['hour'] / 24.0)
    matrix['hour_cos'] = np.cos(2 * np.pi * matrix['hour'] / 24.0)
    matrix.dropna(inplace=True)
    
    # IMPORTANT: Order must strictly match the training data split:
    feature_cols = ['log_return', 'volatility_atr', 'hour_cos', 'hour_sin']
    X_2d = matrix[feature_cols]
    
    if len(X_2d) < LOOKBACK_WINDOW:
        raise ValueError(f"Insufficient historical bars: have {len(X_2d)}, required {LOOKBACK_WINDOW}")

    X_3d = np.array([X_2d.values[-LOOKBACK_WINDOW:]])
    
    return X_2d.tail(1), X_3d, matrix['volatility_atr'].iloc[-1]


def connect_to_mt5() -> bool:
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
    if not mt5.symbol_select(SYMBOL, True):
        print(f"[!] Failed to select {SYMBOL} in Market Watch: {mt5.last_error()}")
        return False

    return True


# Global model instances
xgb_analyst = None
lstm_analyst = None
meta_manager = None


def load_models():
    """Loads and compiles all 3 layers of the AI Boardroom."""
    global xgb_analyst, lstm_analyst, meta_manager
    print("[*] Loading serialized models from local memory cache...")
    
    # 1. XGBoost
    with open("xgb_analyst.pkl", "rb") as f:
        xgb_analyst = pickle.load(f)
    print(" [+] XGBoost Analyst loaded.")
        
    # 2. PyTorch LSTM
    lstm_analyst = ForexLSTMAnalyst(input_dim=4, hidden_dim=64, num_layers=2, output_dim=3)
    lstm_analyst.load_state_dict(torch.load("lstm_analyst.pth", weights_only=True))
    lstm_analyst.eval()
    print(" [+] PyTorch LSTM Analyst loaded.")
        
    # 3. Meta-Manager (Logistic Regression)
    with open("meta_manager.pkl", "rb") as f:
        meta_manager = pickle.load(f)
    print(" [+] Meta-Manager Stack loaded.")


def run_trading_cycle():
    global xgb_analyst, lstm_analyst, meta_manager
    if xgb_analyst is None or lstm_analyst is None or meta_manager is None:
        load_models()

    print(f"\n{'='*60}")
    print(f"[*] Execution cycle triggered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    
    if not connect_to_mt5():
        print("[!] Execution cycle aborted due to connection failure.")
        return

    # Fetch 100 historical hourly bars
    raw_rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS_TO_FETCH)
    account_info = mt5.account_info()
    
    if raw_rates is None or len(raw_rates) == 0:
        print(f"[!] Failed to fetch market rates for {SYMBOL}: {mt5.last_error()}")
        mt5.shutdown()
        return
        
    df = pd.DataFrame(raw_rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
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


if __name__ == "__main__":
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