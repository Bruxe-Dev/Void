"""
Live trading entry point for the new Bidirectional Cascade Engine.

Usage:
    C:\\Users\\HP\\anaconda3\\envs\\void\\python.exe -u scripts/live_cascade_expert.py
"""
import sys
import _bootstrap  # noqa: F401

sys.stdout.reconfigure(line_buffering=True)

from void_fx.live.cascade_runner import main

if __name__ == "__main__":
    main()
