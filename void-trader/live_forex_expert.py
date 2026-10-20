"""
Root compatibility entry point for the legacy v1 live runner.
Delegates to void_fx.live.legacy_runner so all risk safeguards and closed-bar
fixes apply regardless of whether scripts/live_forex_expert.py or root
live_forex_expert.py is invoked.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

sys.stdout.reconfigure(line_buffering=True)

from void_fx.live.legacy_runner import (  # noqa: E402,F401
    ForexLSTMAnalyst,
    load_models,
    main,
    prepare_live_features,
    run_trading_cycle,
)

if __name__ == "__main__":
    main()