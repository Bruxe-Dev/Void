In order to force the logs into a File this will be the new script I'll need to run

# # On Windows PowerShell (Standard in VS Code)
- python -u live_forex_expert.py | Tee-Object -FilePath "trading_logs.txt"