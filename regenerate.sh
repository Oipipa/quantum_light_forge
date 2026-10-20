#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
MPLBACKEND=Agg QLF_EXPORT=1 python main.py
python -m pytest -q
