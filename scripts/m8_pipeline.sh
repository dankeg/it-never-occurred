#!/bin/zsh
# M8 pipeline (owner decisions 2026-10-20): Luna for V-G1 and the three V-G4 studies (shared controls), Sol for V-G5.
# Every run goes through day 5 (end tick 144,000). Resumable: finished runs and study branches are reused on a rerun.
# Schedule, at most five Minecraft servers and no local model: the narrated Sol run starts first and runs throughout;
# the three V-G1 seeds run beside it; then the studies run one after another, four branches at a time.
set -u
cd "$(dirname "$0")/.."
PY=/opt/homebrew/bin/python3.11
LUNA=(--provider chatgpt-plan --model gpt-6-luna)
SOL=(--provider chatgpt-plan --model gpt-6.1-sol --reasoning-effort low)
stage=${1:-all}
log() { echo "$(date '+%m-%d %H:%M') $*"; }

vg1() {  # three 6-day Luna seeds with exact-replay checks (V-G1, V-G2)
  local pids=()
  for spec in "sabotage 1 25700" "sabotage 2 25704" "accident 1 25708"; do
    local parts=(${=spec})
    local variant=${parts[1]} seed=${parts[2]} port=${parts[3]}
    local name=m8_vg1_${variant}_${seed}
    if grep -q "^PASS replay" runs/$name.out 2>/dev/null; then log "$name done"; continue; fi
    $PY scripts/run_village.py --variant $variant $LUNA --days 6 --max-calls 5000 --name $name --heap 3G \
      --base-seed $((20261100 + seed)) --port $port --game-port $((port + 1)) --replay > runs/$name.out 2>&1 < /dev/null &
    pids+=($!)
  done
  for pid in $pids; do wait $pid; done
  for name in m8_vg1_sabotage_1 m8_vg1_sabotage_2 m8_vg1_accident_1; do
    log "$name: $(grep -E '^(PASS|FAIL) replay' runs/$name.out | tail -1)"
  done
}

study() {  # one study on the shared Luna base (the first creates it: factual run, fork and controls)
  $PY scripts/run_study.py --study $1 --variant sabotage --base m8_luna --fork-day 1 --end-day 6 --branches 10 \
    --parallel 4 --heap 3G --port 25712 --game-port 25713 --branch-port 25720 --max-calls 5000 $LUNA \
    > runs/m8_study_$1.out 2>&1 < /dev/null
  log "study $1 exit $?: $(grep -E '"calls_saved_by_gating"' -A 3 runs/m8_study_$1.out | tr '\n' ' ')"
}

study_b() {  # V-G4 redesign (owner, 2026-10-21): no_saboteur and honest_freight fork at agent start, to day 2's end
  $PY scripts/run_study.py --study $1 --variant sabotage --base m8b_luna --fork-day 0 --end-day 3 --branches 10 \
    --parallel 4 --heap 3G --port 25712 --game-port 25713 --branch-port 25800 --max-calls 5000 $LUNA \
    > runs/m8b_study_$1.out 2>&1 < /dev/null
  log "study_b $1 exit $?: $(grep -E '"calls_saved_by_gating"' -A 3 runs/m8b_study_$1.out | tr '\n' ' ')"
}

vg5() {  # the narrated crisis with Sol agents and a Sol GM, scored against the engine shadow
  $PY scripts/run_narrated_village.py --variant sabotage $SOL --ticks 144000 --max-calls 9000 \
    --name m8_narrated_sol --heap 3G --port 25760 --game-port 25761 > runs/m8_narrated_sol.out 2>&1 < /dev/null
  log "narrated exit $?"
}

case $stage in
  vg1) vg1 ;;
  studies) study road_closure; study no_saboteur; study honest_freight ;;
  studies_b) study_b no_saboteur; study_b honest_freight ;;
  vg5) vg5 ;;
  all)
    log "M8 start"
    vg5 &
    vg5_pid=$!
    vg1
    study road_closure; study no_saboteur; study honest_freight
    wait $vg5_pid
    log "M8 runs finished"
    ;;
esac
