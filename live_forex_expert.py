import time
import pickle
import torch
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
LOOKBACK_WINDOW = 24
RISK_PERCENTAGE = 0.01  
MT5_PATH = "C:/Program Files/MetaTrader 5/terminal64.exe"
ACCOUNT_NUMBER = 5057177985

def calculate_position_size(balance: float, atr: float, symbol: str) -> float:
    cash_risk_budget = balance * RISK_PERCENTAGE
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return 0.01 
        
    trade_tick_size = symbol_info.trade_tick_size
    trade_tick_value = symbol_info.trade_tick_value
    volume_step = symbol_info.volume_step
    
    points_per_atr = atr / symbol_info.point
    risk_per_lot = points_per_atr * (trade_tick_value / trade_tick_size) * symbol_info.point
    
    if risk_per_lot == 0:
        return 0.01
        
    raw_lots = cash_risk_budget / risk_per_lot
    
    final_lots = round(raw_lots / volume_step) * volume_step
    return max(symbol_info.volume_min, min(symbol_info.volume_max, final_lots))

def prepare_live_features(df: pd.DataFrame) -> tuple:
    """Transforms raw incoming data into matching 2D and 3D tensor configurations."""
    matrix = df.copy()
    matrix['log_return'] = np.log(matrix['close'] / matrix['close'].shift(1))
    matrix['high_low_span'] = matrix['high'] - matrix['low']
    matrix['volatility_atr'] = matrix['high_low_span'].rolling(window=14).mean()
    matrix['hour'] = matrix.index.hour
    matrix['hour_sin'] = np.sin(2 * np.pi * matrix['hour'] / 24.0)
    matrix['hour_cos'] = np.cos(2 * np.pi * matrix['hour'] / 24.0)
    matrix.dropna(inplace=True)
    
    feature_cols = ['log_return', 'volatility_atr', 'hour_sin', 'hour_cos']
    X_2d = matrix[feature_cols]
    
    X_3d = np.array([X_2d.values[-LOOKBACK_WINDOW:]])
    
    return X_2d.tail(1), X_3d, matrix['volatility_atr'].iloc[-1]

def run_trading_cycle():
    print(f"\nExecution cycle triggered at {datetime.now()}...")
    
    if not mt5.initialize(path=MT5_PATH, login=ACCOUNT_NUMBER):
        print("Live loop failed to connect to MT5.")
        return

    raw_rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 50)
    account_info = mt5.account_info()
    
    if raw_rates is None or account_info is None:
        print("Failed to poll data feeds from server. Retrying next cycle.")
        mt5.shutdown()
        return
        
    df = pd.DataFrame(raw_rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    X_2d_live, X_3d_live, current_atr = prepare_live_features(df)
    
    xgb_probs = xgb_analyst.predict_proba(X_2d_live)
    
    X_3d_tensor = torch.tensor(X_3d_live, dtype=torch.float32)
    with torch.no_grad():
        lstm_logits = lstm_analyst(X_3d_tensor)
        lstm_probs = torch.nn.Softmax(dim=1)(lstm_logits).numpy()
        
    X_blend_live = np.hstack((xgb_probs, lstm_probs))
    
    final_signal = meta_manager.predict(X_blend_live)[0]
    
    print(f"Model Intent Output Class: {final_signal}")
    
    if final_signal == 2:
        live_balance = account_info.balance
        calculated_lots = calculate_position_size(live_balance, current_atr, SYMBOL)
        
        print(f"BUY SIGNAL VALIDATED. Calculated Risk Volume: {calculated_lots} Lots.")
        
        order_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": calculated_lots,
            "type": mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(SYMBOL).ask,
            "sl": mt5.symbol_info_tick(SYMBOL).ask - current_atr,
            "tp": mt5.symbol_info_tick(SYMBOL).ask + (current_atr * 2.0),
            "deviation": 10,
            "magic": 202610,  # Unique tracking tracking handle for your bot
            "comment": "AI TExpert Active Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(order_request)
        print(f"Order Server Response Code: {result.retcode if result else 'FAILED'}")
        
    else:
        print("System evaluates market state as unsafe or flat. Standing down.")
        
    mt5.shutdown()

if __name__ == "__main__":
    print("Booting up AI models from local memory cache...")
    
    with open("xgb_analyst.pkl", "rb") as f:
        xgb_analyst = pickle.load(f)
        
    with open("meta_manager.pkl", "rb") as f:
        meta_manager = pickle.load(f)
        
    
    print("Live Forex TExpert Initialized. Entering 1-Hour Wait Loop...")
    
    while True:
        run_trading_cycle()
        time.sleep(3600)
