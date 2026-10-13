"""
Live trading entry point (legacy v1 models).

    C:\\Users\\HP\\anaconda3\\envs\\void\\python.exe -u scripts/live_forex_expert.py | Tee-Object -FilePath "logs/trading_logs.txt"
"""
import sys

import _bootstrap  # noqa: F401  (adds ./src to sys.path)

# Ensure prints flush immediately in live terminal sessions
sys.stdout.reconfigure(line_buffering=True)

from void_fx.live.legacy_runner import main  # noqa: E402

if __name__ == "__main__":
    main()
