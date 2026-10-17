#!/usr/bin/env bash
# Rebuild every offline result and chart in the README from a clean checkout:
#   python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && bash reproduce.sh
# The live charts (spx_live.png, surface_live.png) depend on the day's quotes and are not rebuilt here.
set -euo pipefail
python tests/data/download.py
python tests/test_rnd.py
python -m rnd --demo --out demo.png
python -m rnd --csv tests/data/spxw_20190626.csv --expiry 2019-07-26 --out spx_2019.png
python -m rnd --csv tests/data/spxw_20190626.csv --surface --out surface.png
