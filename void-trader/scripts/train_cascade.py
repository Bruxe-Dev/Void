"""
train_cascade.py  (v3)
======================
Trains the Bidirectional Two-Stage Cascade and verifies on held-out out-of-sample
data whether the resulting ensemble is profitable after spread/slippage costs.

Key improvements over v2:
  1. Expanded v3 multi-horizon causal feature set (weekly/bi-weekly momentum,
     EMA50/200 distance & slope, 120h range position, day-of-week seasonality,
     realized vol ratio) in `void_fx.features.cascade`.
  2. Exact LSTM 3D sequence window alignment (`t-23 .. t` inclusive) matching
     live inference and `evaluate_h1_baseline.py`.
  3. Mirror-augmented training AND mirror-symmetrised inference for both XGBoost
     and PyTorch LSTM (`void_fx.models.cascade_infer`).
  4. Out-of-fold walk-forward stacking on the training split to train the Stage 2
     Meta-Gate on actual proposed directions without in-sample overfitting.
  5. Single class-balanced uniqueness weighting on the Stage 2 Meta-Gate (fixing
     the v2 double-balanced bug where `class_weight="balanced"` + `balanced_weights`
     forced all probabilities below 0.50), keeping probabilities calibrated around
     0.50 and compatible with `prior_correct` in `evaluate_h1_baseline.py`.
  6. GPU acceleration (`cuda` when available on Colab/Ubuntu, automatic CPU fallback
     for serialized checkpoints).
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import _bootstrap  # noqa: F401
from void_fx.config import (
    ATR_PERIOD,
    CASCADE_MODELS_DIR,
    LEGACY_FEATURE_COLS,
    LEGACY_MODELS_DIR,
    LOOKBACK_WINDOW,
    MAX_HOLDING_BARS,
    MODELS_DIR,
    PT_MULT,
    RAW_DATA_DIR,
    SL_MULT,
    SYMBOLS,
)
from void_fx.features.base import engineer_stationary_features
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
from void_fx.labeling.triple_barrier import (
    OUTCOME_TP,
    bracket_outcomes,
    generate_triple_barrier_labels,
)
from void_fx.models.cascade_infer import (
    CONFIG_FILE,
    meta_features,
    sym_lstm_p_long,
    sym_xgb_p_long,
    windows,
)
from void_fx.models.lstm import ForexLSTMAnalyst

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PURGE = MAX_HOLDING_BARS
SEED = 42


# ============================================================================ data
def load_pair(symbol: str, cost_pips: float) -> pd.DataFrame:
    raw_path = RAW_DATA_DIR / f"{symbol}_H1_30000.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")
    meta_path = RAW_DATA_DIR / f"{symbol}_meta.json"
    point = json.loads(meta_path.read_text(encoding="utf-8"))["point"] if meta_path.exists() else 1e-5
    pip = 10 * point

    raw = pd.read_csv(raw_path, index_col="time", parse_dates=True)
    df = build_cascade_features(raw)
    c = df["close"].to_numpy(float)
    a = df["volatility_atr"].to_numpy(float)

    d = direction_labels(c, a, k=1.0, max_bars=MAX_HOLDING_BARS)
    lo = bracket_outcomes(c, a, side=1, pt_mult=PT_MULT, sl_mult=SL_MULT, max_bars=MAX_HOLDING_BARS)
    sh = bracket_outcomes(c, a, side=-1, pt_mult=PT_MULT, sl_mult=SL_MULT, max_bars=MAX_HOLDING_BARS)

    df["y_dir"] = d["y_dir"].to_numpy()
    df["dir_exit_bar"] = d["exit_bar"].to_numpy()
    df["r_long"], df["r_short"] = lo["r"].to_numpy(), sh["r"].to_numpy()
    df["tp_long"] = (lo["outcome"].to_numpy() == OUTCOME_TP).astype(float)
    df["tp_short"] = (sh["outcome"].to_numpy() == OUTCOME_TP).astype(float)
    df["ex_long"], df["ex_short"] = lo["exit_bar"].to_numpy(), sh["exit_bar"].to_numpy()

    prop_side = np.where(df["y_dir"] == 1.0, 1, -1)
    meta_res = side_aligned_outcome(prop_side, lo, sh)
    df["y_meta"] = meta_res["y_meta"].to_numpy()
    df["meta_exit_bar"] = meta_res["exit_bar"].to_numpy()

    df["cost_r"] = cost_pips * pip / (SL_MULT * df["volatility_atr"])
    df["symbol"] = symbol

    # Match evaluate_h1_baseline.py valid row definition so train/val/test splits align identically
    valid = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=CASCADE_FEATURE_COLS + ["y_dir", "y_meta"]
    ).copy()
    valid["pos"] = np.arange(len(valid))
    return valid


def split_symbol(df: pd.DataFrame):
    """Exact chronological split matching scripts/evaluate_h1_baseline.py:
      - train: first 80%
      - oos  : remaining 20% after 24-bar purge
        - val : first 40% of oos
        - test: last 60% of oos after 24-bar purge
    """
    n = len(df)
    split_idx = int(n * 0.8)
    oos_start = split_idx + PURGE
    oos_len = n - oos_start
    val_end = oos_start + int(oos_len * 0.4)
    test_start = val_end + PURGE
    return (
        np.arange(0, split_idx),
        np.arange(oos_start, val_end),
        np.arange(test_start, n),
    )


def build_split_arrays(dfs, splits, scaler):
    parts = {"train": [], "val": [], "test": []}
    for df, (tr, va, te) in zip(dfs, splits):
        Xn = scaler.transform(df[CASCADE_FEATURE_COLS].to_numpy(float))
        X3 = windows(Xn, LOOKBACK_WINDOW)
        ok = np.arange(len(df)) >= LOOKBACK_WINDOW - 1
        uniq_dir = np.nan_to_num(average_uniqueness(df["dir_exit_bar"].to_numpy(float)), nan=1.0)
        uniq_meta = np.nan_to_num(average_uniqueness(df["meta_exit_bar"].to_numpy(float)), nan=1.0)
        for name, idx in (("train", tr), ("val", va), ("test", te)):
            idx = idx[ok[idx]]
            sub = df.iloc[idx]
            parts[name].append(dict(
                X2=Xn[idx], X3=X3[idx],
                y_dir=sub["y_dir"].to_numpy(float), dir_exit=sub["dir_exit_bar"].to_numpy(float),
                y_meta=sub["y_meta"].to_numpy(float),
                r_long=sub["r_long"].to_numpy(float), r_short=sub["r_short"].to_numpy(float),
                tp_long=sub["tp_long"].to_numpy(float), tp_short=sub["tp_short"].to_numpy(float),
                ex_long=sub["ex_long"].to_numpy(float), ex_short=sub["ex_short"].to_numpy(float),
                cost_r=sub["cost_r"].to_numpy(float),
                time=sub.index.to_numpy(), symbol=sub["symbol"].to_numpy(), pos=sub["pos"].to_numpy(),
                uniq=uniq_dir[idx], uniq_meta=uniq_meta[idx],
            ))
    return {k: {f: np.concatenate([p[f] for p in v]) for f in v[0]} for k, v in parts.items()}


def take(d: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in d.items()}


# ======================================================================= models
def xgb_dir_model(n_estimators, early_stopping=None):
    return XGBClassifier(
        n_estimators=n_estimators, learning_rate=0.025, max_depth=3, min_child_weight=25,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, gamma=0.0,
        tree_method="hist", device=DEVICE, eval_metric="logloss",
        early_stopping_rounds=early_stopping, random_state=SEED, n_jobs=-1,
    )


def fit_xgb_dir(X, y, w, n_estimators=600, Xv=None, yv=None):
    """Mirror-augmented direction XGBoost. Returns (model, best_n_estimators)."""
    Xa, ya, wa = np.vstack([X, mirror_features(X)]), np.concatenate([y, 1 - y]), np.concatenate([w, w])
    if Xv is not None:
        m = xgb_dir_model(n_estimators, early_stopping=60)
        m.fit(Xa, ya, sample_weight=wa,
              eval_set=[(np.vstack([Xv, mirror_features(Xv)]), np.concatenate([yv, 1 - yv]))], verbose=False)
        best = max(40, int(m.best_iteration) + 1)
    else:
        m = xgb_dir_model(n_estimators)
        m.fit(Xa, ya, sample_weight=wa, verbose=False)
        best = n_estimators
    m.set_params(device="cpu")
    return m, best


def fit_lstm_dir(X3, y, w, hidden, layers, max_epochs=20, X3v=None, yv=None, patience=4):
    """Mirror-augmented LSTM with AdamW + early stopping. Returns (model_on_cpu, best_epochs)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = ForexLSTMAnalyst(len(CASCADE_FEATURE_COLS), hidden, 2, layers).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    ce = nn.CrossEntropyLoss(reduction="none")

    Xa = torch.from_numpy(np.concatenate([X3, mirror_features(X3)])).to(DEVICE)
    ya = torch.from_numpy(np.concatenate([y, 1 - y]).astype(np.int64)).to(DEVICE)
    wa = torch.from_numpy(np.concatenate([w, w]).astype(np.float32)).to(DEVICE)
    if X3v is not None:
        Xv = torch.from_numpy(np.concatenate([X3v, mirror_features(X3v)])).to(DEVICE)
        yv_t = torch.from_numpy(np.concatenate([yv, 1 - yv]).astype(np.int64)).to(DEVICE)

    best_loss, best_state, best_epoch, bad = np.inf, None, max_epochs, 0
    n, bs = len(Xa), 512
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = (ce(model(Xa[b]), ya[b]) * wa[b]).sum() / wa[b].sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        msg = f"    epoch {epoch + 1:2d} train_loss={tot / n:.4f}"
        if X3v is not None:
            model.eval()
            with torch.no_grad():
                vl = sum(ce(model(Xv[j:j + 4096]), yv_t[j:j + 4096]).sum().item()
                         for j in range(0, len(Xv), 4096)) / len(Xv)
            msg += f" val_loss={vl:.4f}"
            if vl < best_loss - 1e-4:
                best_loss, best_epoch, bad = vl, epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(msg)
            if bad >= patience:
                print(f"    early stop -> best epoch {best_epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu().eval()
    return model, best_epoch


def fit_meta_gate(Xm_tr, y_tr, w_tr, Xm_va, y_va):
    """Fits both a calibrated L2 LogisticRegression and a shallow XGBoost gate with
    SINGLE class-balanced weights (so output probabilities are centered at 0.50),
    picking whichever achieves higher validation AUC."""
    w_bal = balanced_weights(y_tr.astype(int), base=w_tr)
    w_bal = np.nan_to_num(w_bal, nan=1.0)

    # Candidate A: L2-regularised LogisticRegression on standardised side-aligned features
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)),
    ])
    lr_pipe.fit(Xm_tr, y_tr.astype(int), clf__sample_weight=w_bal)
    auc_lr = roc_auc_score(y_va.astype(int), lr_pipe.predict_proba(Xm_va)[:, 1])

    # Candidate B: Shallow heavily-regularised XGBoost with balanced sample weights
    xgb_g = XGBClassifier(
        n_estimators=400, learning_rate=0.02, max_depth=2, min_child_weight=80,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=15.0,
        tree_method="hist", device=DEVICE, eval_metric="auc",
        early_stopping_rounds=50, random_state=SEED, n_jobs=-1,
    )
    w_va = balanced_weights(y_va.astype(int))
    xgb_g.fit(Xm_tr, y_tr.astype(int), sample_weight=w_bal,
              eval_set=[(Xm_va, y_va.astype(int))], sample_weight_eval_set=[w_va], verbose=False)
    xgb_g.set_params(device="cpu")
    auc_xgb = roc_auc_score(y_va.astype(int), xgb_g.predict_proba(Xm_va)[:, 1])

    print(f"  meta-gate candidates val AUC: LogisticRegression={auc_lr:.4f} | XGBoost={auc_xgb:.4f}")
    if auc_xgb >= auc_lr + 0.005:
        print("  -> selected XGBoost meta-gate")
        return xgb_g, auc_xgb
    print("  -> selected calibrated LogisticRegression meta-gate")
    return lr_pipe, auc_lr


# =================================================================== evaluation
def side_outcomes(d, side):
    long_ = side > 0
    r = np.where(long_, d["r_long"], d["r_short"])
    tp = np.where(long_, d["tp_long"], d["tp_short"])
    ex = np.where(long_, d["ex_long"], d["ex_short"])
    return r, tp, ex


def simulate(take_mask, r_net, exit_bar, symbol, pos, time_):
    """Non-overlapping execution: one open trade per symbol, like the live bot."""
    taken = np.zeros(len(take_mask), bool)
    for sym in np.unique(symbol):
        idx = np.flatnonzero(symbol == sym)
        idx = idx[np.argsort(pos[idx])]
        next_free = -1
        for i in idx:
            if take_mask[i] and pos[i] >= next_free:
                taken[i] = True
                next_free = pos[i] + int(exit_bar[i])
    order = np.argsort(time_[taken], kind="stable")
    rs = r_net[taken][order]
    return taken, rs


def stats(rs: np.ndarray) -> dict:
    if len(rs) == 0:
        return dict(trades=0, win_rate=0.0, avg_r=0.0, total_r=0.0, profit_factor=0.0, max_dd_r=0.0, t_stat=0.0)
    eq = np.cumsum(rs)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq))
    gains, losses = rs[rs > 0].sum(), -rs[rs < 0].sum()
    se = rs.std(ddof=1) / np.sqrt(len(rs)) if len(rs) > 1 else np.nan
    return dict(
        trades=int(len(rs)), win_rate=round(float((rs > 0).mean() * 100), 2),
        avg_r=round(float(rs.mean()), 4), total_r=round(float(rs.sum()), 2),
        profit_factor=round(float(gains / losses), 3) if losses > 0 else float("inf"),
        max_dd_r=round(dd, 2), t_stat=round(float(rs.mean() / se), 2) if se and se > 0 else 0.0,
    )


def fmt(name, s):
    return (f"  {name:28s} trades={s['trades']:5d}  win={s['win_rate']:6.2f}%  avgR={s['avg_r']:+.4f}  "
            f"totalR={s['total_r']:+8.2f}  PF={s['profit_factor']:.3f}  maxDD={s['max_dd_r']:.1f}R  t={s['t_stat']:+.2f}")


def choose_threshold(p, r_net, ex, d, min_trades=60):
    """Pick the gate cutoff in [0.50, 0.65] that maximises risk-adjusted edge (t-stat)
    on validation while requiring at least `min_trades` trades."""
    grid = np.round(np.arange(0.50, 0.65, 0.01), 2)
    scores, counts = [], []
    for t in grid:
        _, rs = simulate(p >= t, r_net, ex, d["symbol"], d["pos"], d["time"])
        st = stats(rs)
        # Require meaningful selectivity and positive edge
        if st["trades"] >= min_trades and st["avg_r"] > 0:
            scores.append(st["t_stat"] * min(st["profit_factor"], 2.0))
        else:
            scores.append(-np.inf)
        counts.append(st["trades"])
    scores = np.array(scores, float)
    finite = np.where(np.isfinite(scores), scores, np.nan)
    smooth = pd.Series(finite).rolling(3, center=True, min_periods=1).mean().to_numpy()
    smooth = np.where(np.isfinite(scores), smooth, -np.inf)
    if not np.isfinite(smooth).any():
        return 0.50, grid, scores, counts
    return float(grid[int(np.nanargmax(smooth))]), grid, scores, counts


def ensure_legacy_models():
    """Ensures `models/legacy/` has the v1 baseline checkpoints so legacy tests and
    baseline comparisons work out of the box on a fresh clone."""
    req = [LEGACY_MODELS_DIR / "xgb_analyst.pkl", LEGACY_MODELS_DIR / "lstm_analyst.pth", LEGACY_MODELS_DIR / "meta_manager.pkl"]
    if all(p.exists() for p in req):
        return
    print("\n[*] Training Legacy v1 baseline models (models/legacy/) for backwards compatibility...")
    LEGACY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(RAW_DATA_DIR / "EURUSD_H1_30000.csv", index_col="time", parse_dates=True)
    feat_df = engineer_stationary_features(raw_df)
    feat_df["target"] = generate_triple_barrier_labels(feat_df, pt_mult=2.0, sl_mult=1.0, max_bars=MAX_HOLDING_BARS)
    clean = feat_df.iloc[:-MAX_HOLDING_BARS].copy()
    split_idx = int(len(clean) * 0.8)
    tr_df = clean.iloc[:split_idx]
    X_tr = tr_df[LEGACY_FEATURE_COLS]
    y_tr = tr_df["target"].map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)

    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight(class_weight="balanced", y=y_tr)
    xgb = XGBClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, objective="multi:softprob", random_state=SEED, tree_method="hist",
    )
    xgb.fit(X_tr, y_tr, sample_weight=sw)

    Xv = X_tr.values.astype(np.float32)
    yv = y_tr.values.astype(np.int64)
    X3 = np.array([Xv[i - LOOKBACK_WINDOW:i] for i in range(LOOKBACK_WINDOW, len(Xv))], dtype=np.float32)
    y3 = yv[LOOKBACK_WINDOW:]

    torch.manual_seed(SEED)
    lstm = ForexLSTMAnalyst(input_dim=4, hidden_dim=64, output_dim=3, num_layers=2).to(DEVICE)
    opt = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    X3_t = torch.from_numpy(X3).to(DEVICE)
    y3_t = torch.from_numpy(y3).to(DEVICE)
    lstm.train()
    for _ in range(8):
        for i in range(0, len(X3_t), 128):
            opt.zero_grad()
            loss = ce(lstm(X3_t[i:i + 128]), y3_t[i:i + 128])
            loss.backward()
            opt.step()
    lstm = lstm.cpu().eval()
    with torch.no_grad():
        lstm_p = torch.softmax(lstm(torch.from_numpy(X3)), dim=1).numpy()
    xgb_p = xgb.predict_proba(X_tr)[LOOKBACK_WINDOW:]
    meta = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=500)
    meta.fit(np.hstack([xgb_p, lstm_p]), y3)

    with open(LEGACY_MODELS_DIR / "xgb_analyst.pkl", "wb") as f:
        pickle.dump(xgb, f)
    with open(LEGACY_MODELS_DIR / "meta_manager.pkl", "wb") as f:
        pickle.dump(meta, f)
    torch.save(lstm.state_dict(), LEGACY_MODELS_DIR / "lstm_analyst.pth")
    print("[+] Legacy v1 models saved to models/legacy/")


# ========================================================================= main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the v3 Bidirectional Cascade (GPU-aware)")
    ap.add_argument("--output-dir", type=Path, default=MODELS_DIR / "cascade_candidate")
    ap.add_argument("--overwrite-baseline", action="store_true", help="write to models/cascade/")
    ap.add_argument("--cost-pips", type=float, default=1.0, help="round-trip spread+slippage per trade (pips)")
    ap.add_argument("--folds", type=int, default=5, help="walk-forward blocks for out-of-fold stacking")
    ap.add_argument("--lstm-hidden", type=int, default=64)
    ap.add_argument("--lstm-layers", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--min-val-trades", type=int, default=50)
    args = ap.parse_args(argv)
    out_dir = CASCADE_MODELS_DIR if args.overwrite_baseline else args.output_dir
    t_start = time.time()
    print(f"[*] Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))

    ensure_legacy_models()

    # ---------------------------------------------------------------- data
    print(f"\n[*] Loading {SYMBOLS} and building v3 features/labels (cost={args.cost_pips} pips)...")
    dfs = [load_pair(s, args.cost_pips) for s in SYMBOLS]
    splits = [split_symbol(df) for df in dfs]
    scaler = FeatureScaler(CASCADE_FEATURE_COLS).fit(
        np.vstack([df.iloc[tr][CASCADE_FEATURE_COLS].to_numpy(float) for df, (tr, _, _) in zip(dfs, splits)]))
    D = build_split_arrays(dfs, splits, scaler)
    tr, va, te = D["train"], D["val"], D["test"]
    for name, d in D.items():
        print(f"    {name:5s}: {len(d['X2']):6,d} rows  {pd.Timestamp(d['time'].min()).date()} -> "
              f"{pd.Timestamp(d['time'].max()).date()}")

    Xtr, X3tr, ytr, wtr = tr["X2"], tr["X3"], tr["y_dir"].astype(int), tr["uniq"]
    Xva, X3va, yva = va["X2"], va["X3"], va["y_dir"].astype(int)
    wtr = wtr / wtr.mean()

    # ------------------------------------------------- stage 1: direction
    print("\n--- STAGE 1: DIRECTION MODELS (fit on 80% train, early-stopped on validation) ---")
    xgb_dir, best_trees = fit_xgb_dir(Xtr, ytr, wtr, Xv=Xva, yv=yva)
    print(f"  XGBoost best trees: {best_trees}")
    print("  LSTM:")
    lstm_dir, best_epochs = fit_lstm_dir(X3tr, ytr, wtr, args.lstm_hidden, args.lstm_layers,
                                         args.max_epochs, X3v=X3va, yv=yva)

    pv_x = sym_xgb_p_long(xgb_dir, va["X2"])
    pv_l = sym_lstm_p_long(lstm_dir, va["X3"])
    blend_scores = {}
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = np.clip(w * pv_x + (1 - w) * pv_l, 1e-6, 1 - 1e-6)
        blend_scores[w] = log_loss(yva, p)
    w_xgb = min(blend_scores, key=blend_scores.get)
    for w, ll in blend_scores.items():
        acc = ((w * pv_x + (1 - w) * pv_l >= 0.5) == yva).mean()
        print(f"  val blend w_xgb={w:.2f}: logloss={ll:.4f} acc={acc * 100:.2f}%" + ("  <- chosen" if w == w_xgb else ""))

    # ---------------------------------- out-of-fold direction probs on train
    print(f"\n--- OUT-OF-FOLD STACKING ({args.folds} walk-forward blocks, purge={PURGE}h) ---")
    t_all = tr["time"].astype("datetime64[ns]")
    edges = np.quantile(t_all.astype(np.int64), np.linspace(0, 1, args.folds + 1)).astype(np.int64).astype("datetime64[ns]")
    oof_x = np.full(len(t_all), np.nan)
    oof_l = np.full(len(t_all), np.nan)
    purge_td = np.timedelta64(PURGE, "h")
    for k in range(1, args.folds):
        pred_mask = (t_all >= edges[k]) & ((t_all < edges[k + 1]) if k < args.folds - 1 else True)
        fit_mask = t_all < (edges[k] - purge_td)
        if fit_mask.sum() < 2000:
            continue
        mx, _ = fit_xgb_dir(Xtr[fit_mask], ytr[fit_mask], wtr[fit_mask], n_estimators=best_trees)
        ml, _ = fit_lstm_dir(X3tr[fit_mask], ytr[fit_mask], wtr[fit_mask], args.lstm_hidden,
                             args.lstm_layers, max_epochs=best_epochs)
        oof_x[pred_mask] = sym_xgb_p_long(mx, tr["X2"][pred_mask])
        oof_l[pred_mask] = sym_lstm_p_long(ml, tr["X3"][pred_mask])
        print(f"  fold {k}: fit={fit_mask.sum():,} predict={pred_mask.sum():,}")
    has_oof = np.isfinite(oof_x)
    p_tr = w_xgb * oof_x + (1 - w_xgb) * oof_l

    # ------------------------------------------------------ stage 2: gate
    print("\n--- STAGE 2: META GATE (calibrated class-balanced risk gate) ---")
    trm = take(tr, has_oof)
    Xm_tr, side_tr = meta_features(trm["X2"], p_tr[has_oof])
    r_tr, tp_tr, _ = side_outcomes(trm, side_tr)

    p_va = w_xgb * pv_x + (1 - w_xgb) * pv_l
    Xm_va, side_va = meta_features(va["X2"], p_va)
    r_va, tp_va, ex_va = side_outcomes(va, side_va)

    gate, val_auc = fit_meta_gate(Xm_tr, tp_tr, trm["uniq_meta"], Xm_va, tp_va)
    g_va = gate.predict_proba(Xm_va)[:, 1]
    print(f"  train TP base rate={tp_tr.mean() * 100:.1f}% | val TP base rate={tp_va.mean() * 100:.1f}% | val AUC={val_auc:.4f}")

    # --------------------------------------------- threshold on validation
    rnet_va = r_va - va["cost_r"]
    thr, grid, scores, counts = choose_threshold(g_va, rnet_va, ex_va, va, args.min_val_trades)
    _, rs = simulate(g_va >= thr, rnet_va, ex_va, va["symbol"], va["pos"], va["time"])
    val_stats = stats(rs)
    _, rs_all = simulate(np.ones(len(g_va), bool), rnet_va, ex_va, va["symbol"], va["pos"], va["time"])
    print(f"  chosen threshold={thr:.2f}")
    print(fmt("VAL  direction only (no gate)", stats(rs_all)))
    print(fmt(f"VAL  gated (p>={thr:.2f})", val_stats))

    # ---------------------------------------------------- held-out test
    print("\n--- HELD-OUT TEST (never used for training or threshold selection) ---")
    pt_x = sym_xgb_p_long(xgb_dir, te["X2"])
    pt_l = sym_lstm_p_long(lstm_dir, te["X3"])
    p_te = w_xgb * pt_x + (1 - w_xgb) * pt_l
    yte = te["y_dir"].astype(int)
    print(f"  direction acc: XGB={((pt_x >= .5) == yte).mean() * 100:.2f}%  "
          f"LSTM={((pt_l >= .5) == yte).mean() * 100:.2f}%  "
          f"Blend={((p_te >= .5) == yte).mean() * 100:.2f}%")
    Xm_te, side_te = meta_features(te["X2"], p_te)
    r_te, tp_te, ex_te = side_outcomes(te, side_te)
    g_te = gate.predict_proba(Xm_te)[:, 1]
    print(f"  gate AUC (chosen dir)={roc_auc_score(tp_te, g_te):.4f} | "
          f"gate AUC (oracle dir)={roc_auc_score(te['y_meta'].astype(int), g_te):.4f}")
    print(classification_report(tp_te.astype(int), (g_te >= 0.50).astype(int),
                                target_names=["Pass/Trap (0)", "Expansion TP (1)"], zero_division=0))
    rnet_te = r_te - te["cost_r"]

    test_report = {}
    rng = np.random.default_rng(SEED)
    rand_side = rng.choice([-1, 1], len(r_te))
    r_rand = side_outcomes(te, rand_side)
    _, rs = simulate(np.ones(len(g_te), bool), r_rand[0] - te["cost_r"], r_rand[2], te["symbol"], te["pos"], te["time"])
    test_report["random_side"] = stats(rs)
    print(fmt("TEST random coin-flip side", test_report["random_side"]))
    _, rs = simulate(np.ones(len(g_te), bool), rnet_te, ex_te, te["symbol"], te["pos"], te["time"])
    test_report["direction_only"] = stats(rs)
    print(fmt("TEST direction only (no gate)", test_report["direction_only"]))
    taken, rs = simulate(g_te >= thr, rnet_te, ex_te, te["symbol"], te["pos"], te["time"])
    test_report["gated"] = stats(rs)
    print(fmt(f"TEST gated (p>={thr:.2f})", test_report["gated"]))
    for sym in SYMBOLS:
        sm = te["symbol"] == sym
        _, rs_s = simulate((g_te >= thr) & sm, rnet_te, ex_te, te["symbol"], te["pos"], te["time"])
        test_report[f"gated_{sym}"] = stats(rs_s)
        print(fmt(f"TEST gated {sym}", test_report[f"gated_{sym}"]))

    g = test_report["gated"]
    approved = bool(val_stats["avg_r"] > 0 and g["trades"] >= 30 and g["avg_r"] > 0 and g["profit_factor"] > 1.1)

    # --------------------------------------------------------------- save
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2), encoding="utf-8")
    with open(out_dir / "xgb_dir.pkl", "wb") as f:
        pickle.dump(xgb_dir, f)
    with open(out_dir / "meta_gate.pkl", "wb") as f:
        pickle.dump(gate, f)
    torch.save({k: v.cpu() for k, v in lstm_dir.state_dict().items()}, out_dir / "lstm_dir.pth")
    config = dict(
        version=3, meta_features="side_aligned_v3", lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers,
        blend_w_xgb=w_xgb, threshold=thr, cost_pips=args.cost_pips, approved=approved,
        trained_at=pd.Timestamp.now(tz="UTC").isoformat(), device=DEVICE,
        xgb_trees=best_trees, lstm_epochs=best_epochs, validation=val_stats, test=test_report,
    )
    (out_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    print(f"\n[+] Saved models + {CONFIG_FILE} to {out_dir}  ({(time.time() - t_start) / 60:.1f} min)")
    print("=" * 100)
    if approved:
        print("[APPROVED] Held-out test is profitable after costs. Next: run scripts/evaluate_h1_baseline.py "
              f"--model-dir {out_dir} and forward-test on a DEMO account.")
    else:
        print("[NOT APPROVED] The held-out test is NOT profitable after costs. The live bot will refuse to trade "
              "these models. Do not deploy them to users.")
    print("=" * 100)


if __name__ == "__main__":
    main()
