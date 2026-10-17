"""
train_cascade.py
================
Trains the Bidirectional Two-Stage Cascade Ensemble:
  1. Direction Analyst (XGBoost + LSTM) on symmetric barriers.
  2. Meta-Label Risk Gate (Calibrated Classifier / Logistic Regression) on proposed 2:1 outcomes.
  3. Uses 24-hour chronological purging gap to ensure absolute data integrity.
  4. Serializes models into `models/cascade/` so live and backtest pipelines can load them safely.
"""

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from void_fx.config import (
    CASCADE_MODELS_DIR,
    LOOKBACK_WINDOW,
    MAX_HOLDING_BARS,
    RAW_DATA_DIR,
    SYMBOLS,
)
from void_fx.features.cascade import (
    CASCADE_FEATURE_COLS,
    FeatureScaler,
    build_cascade_features,
    mirror_features,
)
from void_fx.labeling.cascade import (
    average_uniqueness,
    balanced_weights,
    direction_labels,
    side_aligned_outcome,
)
from void_fx.labeling.triple_barrier import bracket_outcomes
from void_fx.models.lstm import ForexLSTMAnalyst


def load_and_preprocess_pair(symbol: str) -> pd.DataFrame:
    raw_path = RAW_DATA_DIR / f"{symbol}_H1_30000.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")
    raw_df = pd.read_csv(raw_path, index_col="time", parse_dates=True)
    
    # Feature engineering
    df = build_cascade_features(raw_df)
    
    # Label engineering
    c = df["close"].values
    a = df["volatility_atr"].values
    
    dir_res = direction_labels(c, a, k=1.0, max_bars=MAX_HOLDING_BARS)
    long_res = bracket_outcomes(c, a, side=1, pt_mult=2.0, sl_mult=1.0, max_bars=MAX_HOLDING_BARS)
    short_res = bracket_outcomes(c, a, side=-1, pt_mult=2.0, sl_mult=1.0, max_bars=MAX_HOLDING_BARS)
    
    df["y_dir"] = dir_res["y_dir"].values
    df["dir_exit_bar"] = dir_res["exit_bar"].values
    
    # Aligned meta label based on true direction
    prop_side = np.where(df["y_dir"] == 1.0, 1, -1)
    meta_res = side_aligned_outcome(prop_side, long_res, short_res)
    df["y_meta"] = meta_res["y_meta"].values
    df["meta_r"] = meta_res["r"].values
    df["meta_exit_bar"] = meta_res["exit_bar"].values
    
    df["symbol"] = symbol
    return df


def prepare_datasets():
    print("[*] Ingesting and labeling 30,000 bars for EURUSD and GBPUSD...")
    dfs = [load_and_preprocess_pair(s) for s in SYMBOLS]
    
    # We split chronologically per symbol with 24-hour purge
    train_dfs, test_dfs = [], []
    for d in dfs:
        # Drop warm-up and end lookback
        valid = d.dropna(subset=CASCADE_FEATURE_COLS + ["y_dir", "y_meta"]).copy()
        split_idx = int(len(valid) * 0.8)
        tr = valid.iloc[:split_idx].copy()
        te = valid.iloc[split_idx + MAX_HOLDING_BARS:].copy()
        train_dfs.append(tr)
        test_dfs.append(te)
        
    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)
    
    # Fit scaler strictly on training split
    scaler = FeatureScaler(CASCADE_FEATURE_COLS).fit(train_df[CASCADE_FEATURE_COLS].values)
    
    return train_df, test_df, scaler


def build_3d_tensors(X_2d: np.ndarray, y: np.ndarray, lookback: int = 24):
    X_3d, y_3d = [], []
    for i in range(lookback, len(X_2d)):
        X_3d.append(X_2d[i - lookback:i])
        y_3d.append(y[i])
    return np.array(X_3d, dtype=np.float32), np.array(y_3d, dtype=np.int64)


def main():
    train_df, test_df, scaler = prepare_datasets()
    print(f"[+] Total Train Samples: {len(train_df):,} | Test Samples: {len(test_df):,}")
    
    X_train_raw = train_df[CASCADE_FEATURE_COLS].values
    X_test_raw = test_df[CASCADE_FEATURE_COLS].values
    
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    y_train_dir = train_df["y_dir"].values.astype(int)
    y_test_dir = test_df["y_dir"].values.astype(int)
    
    y_train_meta = train_df["y_meta"].values.astype(int)
    y_test_meta = test_df["y_meta"].values.astype(int)
    
    # Mirror augmentation for Direction model: double dataset size and guarantee 50/50 balance
    X_train_mirrored = mirror_features(X_train)
    y_train_dir_mirrored = 1 - y_train_dir
    
    X_train_dir_aug = np.vstack([X_train, X_train_mirrored])
    y_train_dir_aug = np.concatenate([y_train_dir, y_train_dir_mirrored])
    
    # Direction sample weights (uniqueness)
    u_weights = average_uniqueness(train_df["dir_exit_bar"].values)
    u_weights = np.nan_to_num(u_weights, nan=1.0)
    w_dir_aug = np.concatenate([u_weights, u_weights])
    w_dir_aug /= w_dir_aug.mean()
    
    print("\n--- TRAINING STAGE 1: DIRECTION ANALYSTS (BUY VS SELL) ---")
    xgb_dir = XGBClassifier(
        n_estimators=180,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist'
    )
    xgb_dir.fit(X_train_dir_aug, y_train_dir_aug, sample_weight=w_dir_aug)
    
    dir_preds = xgb_dir.predict(X_test)
    dir_acc = accuracy_score(y_test_dir, dir_preds)
    print(f"[+] XGBoost Direction Test Accuracy: {dir_acc * 100:.2f}%")
    print(classification_report(y_test_dir, dir_preds, target_names=["Short (0)", "Long (1)"]))
    
    # LSTM Direction Analyst
    print("[*] Training Bidirectional LSTM Sequential Analyst...")
    X_train_3d, y_train_3d = build_3d_tensors(X_train, y_train_dir, LOOKBACK_WINDOW)
    X_test_3d, y_test_3d = build_3d_tensors(X_test, y_test_dir, LOOKBACK_WINDOW)
    
    torch.manual_seed(42)
    lstm_dir = ForexLSTMAnalyst(
        input_dim=len(CASCADE_FEATURE_COLS),
        hidden_dim=64,
        output_dim=2,  # Binary: 0=Short, 1=Long
        num_layers=2
    )
    
    train_dataset = TensorDataset(torch.tensor(X_train_3d), torch.tensor(y_train_3d))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm_dir.parameters(), lr=0.001)
    
    lstm_dir.train()
    for epoch in range(6):
        loss_val = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = lstm_dir(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            loss_val += loss.item() * len(bx)
        print(f"  Epoch {epoch+1}/6 - Loss: {loss_val / len(train_dataset):.4f}")
        
    lstm_dir.eval()
    with torch.no_grad():
        test_logits = lstm_dir(torch.tensor(X_test_3d))
        lstm_test_preds = torch.argmax(test_logits, dim=1).numpy()
    print(f"[+] LSTM Direction Test Accuracy: {accuracy_score(y_test_3d, lstm_test_preds) * 100:.2f}%")
    
    print("\n--- TRAINING STAGE 2: META-MANAGER RISK GATE (2:1 EXPANSION VS TRAP) ---")
    # Generate probabilities for Stage 2 Meta-Classifier
    xgb_train_p = xgb_dir.predict_proba(X_train)
    xgb_test_p = xgb_dir.predict_proba(X_test)
    
    # Stack predictions with regime/volatility context
    X_meta_train = np.hstack([X_train, xgb_train_p])
    X_meta_test = np.hstack([X_test, xgb_test_p])
    
    w_meta = balanced_weights(y_train_meta, base=average_uniqueness(train_df["meta_exit_bar"].values))
    w_meta = np.nan_to_num(w_meta, nan=1.0)
    
    meta_gate = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    meta_gate.fit(X_meta_train, y_train_meta, sample_weight=w_meta)
    
    meta_preds = meta_gate.predict(X_meta_test)
    meta_probs = meta_gate.predict_proba(X_meta_test)[:, 1]
    
    print(f"[+] Meta-Manager Gate AUC: {roc_auc_score(y_test_meta, meta_probs):.4f}")
    print(classification_report(y_test_meta, meta_preds, target_names=["Pass/Trap (0)", "Expansion TP (1)"]))
    
    # Save cascade models
    CASCADE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CASCADE_MODELS_DIR / "scaler.json", "w") as f:
        json.dump(scaler.to_dict(), f, indent=2)
    with open(CASCADE_MODELS_DIR / "xgb_dir.pkl", "wb") as f:
        pickle.dump(xgb_dir, f)
    with open(CASCADE_MODELS_DIR / "meta_gate.pkl", "wb") as f:
        pickle.dump(meta_gate, f)
    torch.save(lstm_dir.state_dict(), CASCADE_MODELS_DIR / "lstm_dir.pth")
    print("\n[+] All Stage 1 and Stage 2 Cascade Models serialized successfully to models/cascade/")


if __name__ == "__main__":
    main()
