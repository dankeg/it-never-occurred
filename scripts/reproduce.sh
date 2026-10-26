#!/bin/sh
# Recompute every number, table and figure in the paper from the shipped logs, then rebuild the PDF.
#
# $0 and offline: judge verdicts are read from the caches shipped with each run, so no model is called and no
# sign-in is needed. Takes a few minutes. Afterwards, scripts/check_reproduction.py compares every output with the
# committed version.
#
# Usage: sh scripts/reproduce.sh            (uses `python3`; set PYTHON=... to choose another interpreter;
#        Python 3.11 reproduces every value exactly, see the README)
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}

[ -d runs/jub_A_1 ] || "$PY" scripts/unpack_logs.py

echo "== Counterfactual branches at noon on day 1 (Section 4.4, Figure 3, Table 4), from cached verdicts"
# Runs in the order they were judged for the paper, so that the rows come out in the committed order.
BRANCHED="runs/jub_A_[0-9] runs/jub_L3_[0-9] runs/jub_D_[0-9] runs/jub_N3_[0-9] runs/jub_V2p_[0-9] runs/jub_V2o_[0-9]"
for variant in asked jubilee candid self drive duty duty_jubilee counsellor; do
  "$PY" scripts/jubilee_asked.py judge $BRANCHED --variant "$variant" > /dev/null
done

echo "== Experiment 2: when the mayor first suspected (Section 4.5)"
"$PY" scripts/jubilee_v2_suspicion.py runs/jub_V2p_[0-9] runs/jub_V2t_[0-9] runs/jub_V2x_[0-9] runs/jub_V2o_[0-9] > /dev/null

echo "== Night crossings told of the watchman's patrol (Section 4.2)"
"$PY" scripts/jubilee_crossings.py report runs/jub_V2p_[0-9] > /dev/null

echo "== Recruitment of outsiders and the fresh-bread episodes (Section 4.1, Table 1)"
"$PY" scripts/jubilee_recruitment.py > /dev/null
"$PY" scripts/jubilee_recruitment.py --fresh > /dev/null

echo "== Experiment 1 and Experiment 2 analyses (every count, d' and c in the paper)"
"$PY" scripts/jubilee_analysis.py > /dev/null
"$PY" scripts/jubilee_v2_analysis.py > /dev/null

echo "== Figures and thumbnail"
"$PY" scripts/jubilee_figures.py
"$PY" scripts/jubilee_thumbnail.py > /dev/null

if command -v pandoc > /dev/null && command -v xelatex > /dev/null; then
  echo "== Tables and the PDF (docs/report/report.pdf)"
  "$PY" scripts/jubilee_report_build.py > /dev/null
else
  echo "== Tables only (pandoc or xelatex not found, so the PDF is not rebuilt)"
  "$PY" scripts/jubilee_report_build.py --tables-only
fi

echo
"$PY" scripts/check_reproduction.py
