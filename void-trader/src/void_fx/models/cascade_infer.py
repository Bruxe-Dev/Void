"""
Shared cascade inference (v3).

Training, evaluation and live trading ALL call `CascadeModel.predict`, so the
exact same feature windows, mirror-symmetrisation and meta features are used
everywhere. (Before v3 each of those three places had its own copy of the math,
and they had drifted apart: e.g. the LSTM was trained on windows that excluded
the current bar but run live on windows that included it.)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from void_fx.config import LOOKBACK_WINDOW
from void_fx.features.cascade import (
    ANTISYMMETRIC_COLS,
    CASCADE_FEATURE_COLS,
    FeatureScaler,
    mirror_features,
)
from void_fx.models.lstm import ForexLSTMAnalyst

CONFIG_FILE = "cascade_config.json"

# Used when a checkpoint has no cascade_config.json (pre-v3 models).
DEFAULT_CONFIG = dict(
    version=2,
    lstm_hidden=64,
    lstm_layers=2,
    blend_w_xgb=0.5,
    threshold=0.50,
    meta_features="legacy",
    approved=False,
)


def windows(X_norm: np.ndarray, lookback: int = LOOKBACK_WINDOW) -> np.ndarray:
    """Row i -> X_norm[i-lookback+1 : i+1] (INCLUDES bar i, exactly like live).
    The first lookback-1 rows are zero-padded; callers must ignore them."""
    X_norm = np.asarray(X_norm, np.float32)
    n, f = X_norm.shape
    out = np.zeros((n, lookback, f), np.float32)
    if n >= lookback:
        v = np.lib.stride_tricks.sliding_window_view(X_norm, lookback, axis=0)  # (n-lb+1, f, lb)
        out[lookback - 1:] = v.transpose(0, 2, 1)
    return out


def sym_xgb_p_long(xgb, X2d: np.ndarray, feature_cols=CASCADE_FEATURE_COLS) -> np.ndarray:
    """Mirror-symmetrised P(long): average of the chart and the upside-down chart."""
    p = xgb.predict_proba(X2d)[:, 1]
    pm = xgb.predict_proba(mirror_features(X2d, feature_cols))[:, 1]
    return 0.5 * (p + 1.0 - pm)


@torch.no_grad()
def sym_lstm_p_long(lstm, X3d: np.ndarray, device: str = "cpu", batch: int = 4096,
                    feature_cols=CASCADE_FEATURE_COLS) -> np.ndarray:
    lstm.eval()

    def run(X):
        outs = []
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(device)
            outs.append(torch.softmax(lstm(xb), dim=1)[:, 1].float().cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros(0, np.float32)

    return 0.5 * (run(X3d) + 1.0 - run(mirror_features(X3d, feature_cols)))


def meta_features(X2d: np.ndarray, p_long: np.ndarray, feature_cols=CASCADE_FEATURE_COLS):
    """
    Side-aligned meta features. Antisymmetric features are multiplied by the
    proposed side, so the gate sees the market from the TRADE's point of view
    (a strong down-move is 'favourable momentum' for a short). Adds direction
    confidence and the side itself.
    Returns (X_meta, side) with side in {+1, -1}.
    """
    p_long = np.asarray(p_long, float)
    side = np.where(p_long >= 0.5, 1.0, -1.0)
    Xa = np.array(X2d, np.float32, copy=True)
    anti_idx = [feature_cols.index(c) for c in ANTISYMMETRIC_COLS if c in feature_cols]
    Xa[:, anti_idx] *= side[:, None].astype(np.float32)
    conf = (np.abs(p_long - 0.5) * 2.0)[:, None]
    return np.hstack([Xa, conf, side[:, None]]).astype(np.float32), side.astype(int)


class CascadeModel:
    """Bundle of scaler + direction models + meta gate + config."""

    def __init__(self, scaler: FeatureScaler, xgb, lstm, meta, config: dict, device: str = "cpu"):
        self.scaler, self.xgb, self.lstm, self.meta = scaler, xgb, lstm, meta
        self.config = config
        self.device = device
        self.feature_cols = list(scaler.feature_cols)  # each checkpoint carries its own feature list

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, model_dir: Path, device: str = "cpu") -> "CascadeModel":
        model_dir = Path(model_dir)
        cfg = dict(DEFAULT_CONFIG)
        cfg_path = model_dir / CONFIG_FILE
        if cfg_path.exists():
            cfg.update(json.loads(cfg_path.read_text(encoding="utf-8")))

        scaler = FeatureScaler.from_dict(json.loads((model_dir / "scaler.json").read_text(encoding="utf-8")))
        with open(model_dir / "xgb_dir.pkl", "rb") as f:
            xgb = pickle.load(f)
        with open(model_dir / "meta_gate.pkl", "rb") as f:
            meta = pickle.load(f)
        for m in (xgb, meta):  # models trained on a Colab GPU must run on CPU machines
            if hasattr(m, "get_xgb_params"):
                try:
                    m.set_params(device="cpu")
                except Exception:
                    pass

        lstm = ForexLSTMAnalyst(
            input_dim=len(scaler.feature_cols),
            hidden_dim=int(cfg["lstm_hidden"]),
            output_dim=2,
            num_layers=int(cfg["lstm_layers"]),
        )
        lstm.load_state_dict(torch.load(model_dir / "lstm_dir.pth", map_location="cpu", weights_only=True))
        lstm.to(device).eval()
        return cls(scaler, xgb, lstm, meta, cfg, device)

    # ------------------------------------------------------------- predict
    @property
    def threshold(self) -> float:
        return float(self.config["threshold"])

    @property
    def approved(self) -> bool:
        return bool(self.config.get("approved", False))

    def predict_norm(self, X_norm: np.ndarray) -> dict:
        """X_norm: scaled features for consecutive CLOSED bars of ONE symbol.
        Returns per-row arrays. Rows < LOOKBACK_WINDOW-1 have no full LSTM window."""
        X_norm = np.asarray(X_norm, np.float32)
        X3 = windows(X_norm)
        fc = self.feature_cols
        pl_x = sym_xgb_p_long(self.xgb, X_norm, fc)
        pl_l = sym_lstm_p_long(self.lstm, X3, self.device, feature_cols=fc)
        w = float(self.config["blend_w_xgb"])
        p_long = w * pl_x + (1.0 - w) * pl_l

        if self.config.get("meta_features") == "legacy":
            X_meta = np.hstack([X_norm, np.column_stack([1.0 - pl_x, pl_x])])
            side = np.where(p_long >= 0.5, 1, -1)
        else:
            X_meta, side = meta_features(X_norm, p_long, fc)
        p_gate = self.meta.predict_proba(X_meta)[:, 1]
        return dict(
            p_long_xgb=pl_x,
            p_long_lstm=pl_l,
            p_long=p_long,
            direction=side,
            p_expansion=p_gate,
            warmup_mask=np.arange(len(X_norm)) >= (LOOKBACK_WINDOW - 1),
        )

    def predict_raw(self, X_raw: np.ndarray) -> dict:
        """X_raw columns must be in self.feature_cols order."""
        return self.predict_norm(self.scaler.transform(X_raw))

    def predict_frame(self, feats_df) -> dict:
        """feats_df: output of build_cascade_features (rows must be finite for self.feature_cols)."""
        return self.predict_raw(feats_df[self.feature_cols].to_numpy(dtype=float))
