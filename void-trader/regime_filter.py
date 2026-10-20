"""
Root compatibility shim re-exporting void_fx.features.regime.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from void_fx.features.regime import (  # noqa: E402,F401
    ADX_ENTER_THRESHOLD,
    ADX_EXIT_THRESHOLD,
    ADX_PERIOD,
    HURST_MAX_LAG,
    HURST_THRESHOLD,
    HURST_WINDOW,
    MIN_CONVERGENCE_BARS,
    USE_HURST_CONFIRMATION,
    RegimeDecision,
    _hysteresis_state,
    _wilder_smooth,
    compute_adx,
    compute_regime_features,
    compute_rolling_hurst,
    evaluate_live_regime,
    regime_diagnostics,
)