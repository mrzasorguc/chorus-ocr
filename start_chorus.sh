#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.10-3.13 is required."
  exit 1
fi
if ! command -v tesseract >/dev/null 2>&1; then
  echo "Tesseract is required for Standard and Maximum modes."
  echo "Install it with your system package manager, then run this file again."
  exit 1
fi

printf "Install GOT-OCR? It is required for Maximum Performance mode. [y/N]: "
read -r install_got

if [ ! -x ".venv/bin/python" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
if [[ "$install_got" =~ ^[Yy]$ ]]; then
  .venv/bin/python -m pip install -e ".[demo,got]"
else
  .venv/bin/python -m pip install -e ".[demo]"
fi
.venv/bin/python -m chorus.web
