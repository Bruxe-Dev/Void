#!/usr/bin/env bash
# Train Void Trader on a Colab GPU and pull the models back.
#   usage: scripts/colab_train.sh [GPU=T4] [extra train_cascade.py args...]
# Results land in models/cascade_candidate/ and models/legacy/ locally.
set -euo pipefail
GPU="${1:-T4}"; shift || true
SESSION="void-train"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAR="/tmp/void-trader.tgz"

echo "[*] Packing project..."
tar czf "$TAR" --exclude=.git --exclude=.venv --exclude='__pycache__' --exclude='*.egg-info' \
    --exclude='models/cascade_candidate' -C "$(dirname "$ROOT")" "$(basename "$ROOT")"

echo "[*] Starting Colab session ($GPU)..."
colab new -s "$SESSION" --gpu "$GPU" || colab status -s "$SESSION"
colab upload -s "$SESSION" "$TAR" /content/void-trader.tgz

cat > /tmp/void_remote.py <<EOF
import subprocess
cmd = r'''
set -e
cd /content && rm -rf void-trader && tar xzf void-trader.tgz && cd void-trader
pip install -q -r requirements.txt && pip install -q -e . --no-deps
python -u scripts/train_cascade.py $*
tar czf /content/trained_models.tgz models/cascade_candidate models/legacy
'''
p = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(p.stdout[-20000:])
if p.returncode != 0:
    raise RuntimeError(f"Remote training exited with code {p.returncode}")
EOF
colab exec -s "$SESSION" -f /tmp/void_remote.py | tee "$ROOT/colab_train.log"

echo "[*] Downloading models..."
colab download -s "$SESSION" /content/trained_models.tgz /tmp/trained_models.tgz
tar xzf /tmp/trained_models.tgz -C "$ROOT"
colab stop -s "$SESSION"
echo "[+] Done. Models in models/cascade_candidate/ and models/legacy/, log in colab_train.log"
