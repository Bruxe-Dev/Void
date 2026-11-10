"""
evaluate_h1_baseline.py
=======================
Reproducible evaluation, account-size-aware backtesting ($50, $100, $500, $100,000),
and disciplined experiments against the unchanged H1 Cascade and Legacy baselines.

Usage:
    python scripts/evaluate_h1_baseline.py --model-dir models/cascade_candidate
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
from sklearn.metrics import accuracy_score, roc_auc_score

import _bootstrap  # noqa: F401
from void_fx.backtest.engine import BacktestConfig, run_backtest
from void_fx.config import (
    CASCADE_MODELS_DIR,
    LEGACY_FEATURE_COLS,
    LEGACY_MODELS_DIR,
    LOOKBACK_WINDOW,
    MAX_HOLDING_BARS,
    RAW_DATA_DIR,
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
    breakeven_probability,
    direction_labels,
    prior_correct,
    side_aligned_outcome,
)
from void_fx.labeling.triple_barrier import bracket_outcomes, generate_triple_barrier_labels
from void_fx.models.cascade_infer import CascadeModel
from void_fx.models.lstm import ForexLSTMAnalyst


def load_cascade_checkpoint(model_dir: Path = CASCADE_MODELS_DIR) -> CascadeModel:
    """Loads scaler + models + cascade_config.json through the shared CascadeModel."""
    return CascadeModel.load(model_dir)


def prepare_symbol_splits(symbol: str):
    raw_path = RAW_DATA_DIR / f"{symbol}_H1_30000.csv"
    meta_path = RAW_DATA_DIR / f"{symbol}_meta.json"
    raw_df = pd.read_csv(raw_path, index_col="time", parse_dates=True)
    point = json.loads(meta_path.read_text(encoding="utf-8"))["point"]

    df = build_cascade_features(raw_df)
    c = df["close"].to_numpy(dtype=float)
    a = df["volatility_atr"].to_numpy(dtype=float)

    dir_res = direction_labels(c, a, k=1.0, max_bars=MAX_HOLDING_BARS)
    long_res = bracket_outcomes(c, a, side=1, pt_mult=2.0, sl_mult=1.0, max_bars=MAX_HOLDING_BARS)
    short_res = bracket_outcomes(c, a, side=-1, pt_mult=2.0, sl_mult=1.0, max_bars=MAX_HOLDING_BARS)

    df["y_dir"] = dir_res["y_dir"].to_numpy()
    df["r_long"] = long_res["r"].to_numpy()
    df["r_short"] = short_res["r"].to_numpy()
    df["out_long"] = long_res["outcome"].to_numpy()
    df["out_short"] = short_res["outcome"].to_numpy()

    prop_side = np.where(df["y_dir"] == 1.0, 1, -1)
    meta_res = side_aligned_outcome(prop_side, long_res, short_res)
    df["y_meta"] = meta_res["y_meta"].to_numpy()

    valid = df.dropna(subset=CASCADE_FEATURE_COLS + ["y_dir", "y_meta"]).copy()
    split_idx = int(len(valid) * 0.8)
    train_df = valid.iloc[:split_idx].copy()
    oos_df = valid.iloc[split_idx + MAX_HOLDING_BARS :].copy()

    # Split OOS into chronological validation (first 40%) and held-out test (last 60% after 24-bar purge)
    val_end = int(len(oos_df) * 0.4)
    val_df = oos_df.iloc[:val_end].copy()
    test_df = oos_df.iloc[val_end + MAX_HOLDING_BARS :].copy()
    return train_df, val_df, test_df, oos_df, point


def compute_cascade_predictions(df: pd.DataFrame, model: CascadeModel, symmetrize: bool = True) -> dict:
    """Vectorized causal inference through the SAME code path as live trading.
    (`symmetrize` is kept for backwards compatibility; v3 inference is always symmetrised.)"""
    t0 = time.perf_counter()
    out = model.predict_frame(df)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    px, pl = out["p_long_xgb"], out["p_long_lstm"]
    return dict(
        xgb_probs=np.column_stack([1 - px, px]),
        lstm_probs=np.column_stack([1 - pl, pl]),
        p_long=out["p_long"],
        p_short=1.0 - out["p_long"],
        direction=out["direction"],
        p_expansion=out["p_expansion"],
        warmup_mask=out["warmup_mask"],
        elapsed_ms=elapsed_ms,
    )


def evaluate_legacy_baseline(symbol: str = "EURUSD"):
    raw_df = pd.read_csv(RAW_DATA_DIR / f"{symbol}_H1_30000.csv", index_col="time", parse_dates=True)
    feat_df = engineer_stationary_features(raw_df)
    labels = generate_triple_barrier_labels(feat_df, pt_mult=2.0, sl_mult=1.0, max_bars=MAX_HOLDING_BARS)
    feat_df["target"] = labels
    feat_df = feat_df.iloc[:-MAX_HOLDING_BARS].copy()

    split_idx = int(len(feat_df) * 0.8)
    test_df = feat_df.iloc[split_idx + MAX_HOLDING_BARS :].copy()

    with open(LEGACY_MODELS_DIR / "xgb_analyst.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(LEGACY_MODELS_DIR / "meta_manager.pkl", "rb") as f:
        meta = pickle.load(f)
    lstm = ForexLSTMAnalyst(input_dim=4, hidden_dim=64, num_layers=2, output_dim=3)
    lstm.load_state_dict(torch.load(LEGACY_MODELS_DIR / "lstm_analyst.pth", weights_only=True))
    lstm.eval()

    X_2d = test_df[LEGACY_FEATURE_COLS].to_numpy(dtype=np.float32)
    n = len(X_2d)
    X_3d = np.array([X_2d[i - LOOKBACK_WINDOW + 1 : i + 1] for i in range(LOOKBACK_WINDOW - 1, n)], dtype=np.float32)
    xgb_p = xgb.predict_proba(X_2d[LOOKBACK_WINDOW - 1 :])
    with torch.no_grad():
        lstm_p = torch.softmax(lstm(torch.tensor(X_3d)), dim=1).numpy()
    X_blend = np.hstack([xgb_p, lstm_p])
    preds = meta.predict(X_blend)

    signals = np.zeros(n, dtype=int)
    signals[LOOKBACK_WINDOW - 1 :] = np.where(preds == 2, 1, 0)
    raw_aligned = raw_df.loc[test_df.index].copy()
    raw_aligned["volatility_atr"] = test_df["volatility_atr"]
    bt = run_backtest(raw_aligned, signals, symbol=symbol, cfg=BacktestConfig(starting_balance=100_000.0))
    return dict(
        symbol=symbol,
        test_bars=n,
        buy_signals=int((signals == 1).sum()),
        accepted_trades=bt.accepted_trades,
        win_rate=bt.win_rate,
        return_pct=bt.return_pct,
        max_dd_pct=bt.max_drawdown_pct,
        signal_mean_r=bt.signal_only_mean_r,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate H1 baseline and account-aware backtests")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to save JSON summary")
    parser.add_argument("--model-dir", type=Path, default=CASCADE_MODELS_DIR, help="Cascade checkpoint directory")
    args = parser.parse_args()

    model = load_cascade_checkpoint(args.model_dir)
    thr = model.threshold
    print(f"[*] Model dir: {args.model_dir} | version={model.config.get('version')} | threshold={thr:.2f} "
          f"| approved={model.approved}")
    print("=" * 78)
    print("VOID TRADER — H1 BASELINE VERIFICATION & ACCOUNT-AWARE EVALUATION")
    print("=" * 78)

    summary = {"predictive": {}, "account_scenarios": {}, "experiments": {}, "legacy": {}}

    # 1. Legacy v1 baseline check on EURUSD
    try:
        leg = evaluate_legacy_baseline("EURUSD")
    except FileNotFoundError as e:
        leg = None
        print(f"\n[1] Legacy v1 models not found, skipping ({e.filename})")
    if leg:
        summary["legacy"]["EURUSD"] = leg
        print(f"\n[1] Legacy v1 (Long-Only 3-Class) OOS on EURUSD ({leg['test_bars']} bars):")
        print(f"    Signals={leg['buy_signals']} | Trades={leg['accepted_trades']} | WinRate={leg['win_rate']:.2f}% "
              f"| Return($100k)={leg['return_pct']:+.2f}% | MaxDD={leg['max_dd_pct']:.2f}% | Signal Mean R={leg['signal_mean_r']:+.4f}")

    # 2. Bidirectional Cascade H1 Baseline predictive & trading evaluation
    print("\n[2] Bidirectional Cascade H1 Baseline — Out-of-Sample Predictive Metrics:")
    splits = {}
    for sym in SYMBOLS:
        tr, va, te, oos, pt = prepare_symbol_splits(sym)
        splits[sym] = (tr, va, te, oos, pt)
        pred = compute_cascade_predictions(oos, model)
        m = pred["warmup_mask"]
        y_dir = oos["y_dir"].to_numpy()[m].astype(int)
        xgb_acc = accuracy_score(y_dir, (pred["xgb_probs"][m, 1] >= 0.5).astype(int))
        lstm_acc = accuracy_score(y_dir, (pred["lstm_probs"][m, 1] >= 0.5).astype(int))
        blend_acc = accuracy_score(y_dir, (pred["direction"][m] == 1).astype(int))

        # True outcome in the MODEL's proposed direction (not oracle direction)
        actual_tp_in_chosen_dir = np.where(
            pred["direction"][m] == 1,
            oos["out_long"].to_numpy()[m] == 1,
            oos["out_short"].to_numpy()[m] == 1,
        ).astype(int)
        meta_auc_oracle = roc_auc_score(oos["y_meta"].to_numpy()[m].astype(int), pred["p_expansion"][m])
        meta_auc_chosen = roc_auc_score(actual_tp_in_chosen_dir, pred["p_expansion"][m])

        summary["predictive"][sym] = dict(
            train_range=f"{tr.index[0]} -> {tr.index[-1]} ({len(tr)} bars)",
            oos_range=f"{oos.index[0]} -> {oos.index[-1]} ({len(oos)} bars)",
            xgb_dir_acc=round(xgb_acc * 100, 2),
            lstm_dir_acc=round(lstm_acc * 100, 2),
            blend_dir_acc=round(blend_acc * 100, 2),
            meta_auc_oracle=round(float(meta_auc_oracle), 4),
            meta_auc_chosen=round(float(meta_auc_chosen), 4),
            inference_ms_total=round(pred["elapsed_ms"], 2),
        )
        print(f"  {sym} ({oos.index[0]} -> {oos.index[-1]}, n={ m.sum() }):")
        print(f"    Direction Acc: XGB={xgb_acc*100:.2f}% | LSTM={lstm_acc*100:.2f}% | Blended={blend_acc*100:.2f}%")
        print(f"    Meta-Gate AUC (oracle dir)={meta_auc_oracle:.4f} | (model dir)={meta_auc_chosen:.4f} "
              f"| Base TP rate={actual_tp_in_chosen_dir.mean()*100:.2f}% | Inference={pred['elapsed_ms']:.1f}ms")

    # 3. Account-size-aware backtests on EURUSD OOS for $50, $100, $500, $100,000
    print("\n[3] Account-Size-Aware Risk Manager Backtest on EURUSD OOS (Baseline H1 Cascade):")
    _, _, _, eurusd_oos, eurusd_pt = splits["EURUSD"]
    pred_eu = compute_cascade_predictions(eurusd_oos, model)
    base_signals_eu = np.where(
        pred_eu["warmup_mask"] & (pred_eu["p_expansion"] >= thr),
        pred_eu["direction"],
        0,
    )
    for bal in (50.0, 100.0, 500.0, 100_000.0):
        bt = run_backtest(eurusd_oos, base_signals_eu, symbol="EURUSD", point=eurusd_pt,
                          cfg=BacktestConfig(starting_balance=bal))
        summary["account_scenarios"][str(int(bal))] = dict(
            starting_balance=bal,
            ending_balance=bt.ending_balance,
            return_pct=bt.return_pct,
            max_dd_pct=bt.max_drawdown_pct,
            total_signals=bt.total_signals,
            accepted_trades=bt.accepted_trades,
            rejected_orders=bt.rejected_orders,
            avg_lots=bt.avg_lots,
            min_lots=bt.min_lots,
            max_lots=bt.max_lots,
            avg_risk_money=bt.avg_planned_risk_money,
            avg_risk_pct=bt.avg_planned_risk_pct,
            peak_margin_pct=bt.peak_Margin_utilization_pct,
            halted=bt.halted,
            rejection_reasons=bt.rejection_reasons,
        )
        print(f"  Balance=${bal:>10,.2f} -> End=${bt.ending_balance:>10,.2f} ({bt.return_pct:+6.2f}%) "
              f"| MaxDD={bt.max_drawdown_pct:5.2f}% | Accepted={bt.accepted_trades:3d} | Rejected={bt.rejected_orders:4d} "
              f"| Lots=[{bt.min_lots:.2f}..{bt.max_lots:.2f}] (avg {bt.avg_lots:.2f}) "
              f"| AvgRisk=${bt.avg_planned_risk_money:.2f} ({bt.avg_planned_risk_pct:.2f}%) | Halted={bt.halted}")
        print(f"    Rejection breakdown: {bt.rejection_reasons}")

    # 4. Disciplined experiments on Validation vs Held-Out Test (both symbols)
    print("\n[4] Disciplined Experiments (Validation Selection -> Held-Out Test Verification):")
    exp_defs = [
        ("E0_Model_Threshold", dict(symmetrize=True, thr=thr, use_prior=False, adx_min=None, di_agree=False)),
        ("E1_MirrorSymmetrized_p050", dict(symmetrize=True, thr=0.50, use_prior=False, adx_min=None, di_agree=False)),
        ("E2_MetaThreshold_p054", dict(symmetrize=True, thr=0.54, use_prior=False, adx_min=None, di_agree=False)),
        ("E3_PriorCalibrated_Breakeven", dict(symmetrize=True, thr=0.0, use_prior=True, adx_min=None, di_agree=False)),
        ("E4_RegimeFilter_ADX20", dict(symmetrize=True, thr=0.50, use_prior=False, adx_min=20.0, di_agree=False)),
        ("E5_HTF_DI_Agreement", dict(symmetrize=True, thr=0.50, use_prior=False, adx_min=None, di_agree=True)),
    ]

    for exp_name, params in exp_defs:
        val_r, test_r = [], []
        val_trades, test_trades = 0, 0
        test_ret_sum, test_dd_max = 0.0, 0.0
        for sym in SYMBOLS:
            tr, va, te, _, pt = splits[sym]
            # Base rate of y_meta on training set for prior correction
            prior_tp = float(tr["y_meta"].mean())
            for part_name, part_df in (("val", va), ("test", te)):
                pr = compute_cascade_predictions(
                    part_df, model, symmetrize=params["symmetrize"]
                )
                active = pr["warmup_mask"].copy()
                if params["use_prior"]:
                    p_true = prior_correct(pr["p_expansion"], prior=prior_tp)
                    sp_pts = np.where(part_df["spread"].to_numpy() > 0, part_df["spread"].to_numpy(), 2.0)
                    cost_r = (sp_pts * pt) / part_df["volatility_atr"].to_numpy()
                    be_p = breakeven_probability(cost_r, pt_over_sl=2.0)
                    active &= p_true >= (be_p + 0.01)
                else:
                    active &= pr["p_expansion"] >= params["thr"]

                if params["adx_min"] is not None:
                    active &= (part_df["adx_n"].to_numpy() * 100.0) >= params["adx_min"]
                if params["di_agree"]:
                    active &= (pr["direction"] * part_df["di_spread"].to_numpy()) > 0

                sigs = np.where(active, pr["direction"], 0)
                bt = run_backtest(part_df, sigs, symbol=sym, point=pt, cfg=BacktestConfig(starting_balance=100_000.0))
                if part_name == "val":
                    val_trades += bt.accepted_trades
                    val_r.extend([t.realized_r for t in bt.trades])
                else:
                    test_trades += bt.accepted_trades
                    test_r.extend([t.realized_r for t in bt.trades])
                    test_ret_sum += bt.return_pct
                    test_dd_max = max(test_dd_max, bt.max_drawdown_pct)

        val_mean_r = float(np.mean(val_r)) if val_r else 0.0
        test_mean_r = float(np.mean(test_r)) if test_r else 0.0
        test_se = float(np.std(test_r, ddof=1) / np.sqrt(len(test_r))) if len(test_r) > 1 else float("nan")
        test_t = test_mean_r / test_se if (test_se and np.isfinite(test_se) and test_se > 0) else 0.0
        summary["experiments"][exp_name] = dict(
            val_trades=val_trades,
            val_mean_r=round(val_mean_r, 4),
            test_trades=test_trades,
            test_mean_r=round(test_mean_r, 4),
            test_t_stat=round(test_t, 2),
            test_combined_return_pct=round(test_ret_sum, 2),
            test_max_dd_pct=round(test_dd_max, 2),
        )
        print(f"  {exp_name:30s} | Val: n={val_trades:3d}, E[R]={val_mean_r:+.4f} "
              f"| Test: n={test_trades:3d}, E[R]={test_mean_r:+.4f} (t={test_t:+.2f}), "
              f"CombRet={test_ret_sum:+6.2f}%, MaxDD={test_dd_max:5.2f}%")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[+] Saved evaluation report to {args.json_out}")


if __name__ == "__main__":
    main()
