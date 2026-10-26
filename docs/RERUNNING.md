# Rerunning the experiments

Levels 1 and 2 of the [README](../README.md) recompute every result from the shipped logs at no cost. This page is
for rerunning the simulation itself: replaying a shipped episode exactly, running new episodes and branches, and
judging them. It needs a Minecraft server and, for anything beyond a replay, a model.

Rerun in a fresh clone and do not unpack the shipped logs there: runners refuse to overwrite an existing run, and the
analysis scripts read whatever is in `runs/`.

## Requirements

- **macOS on one APFS volume.** The harness gives every run a copy-on-write clone of the prepared world, and `runs/`
  must be on the same APFS volume as the fixtures (clones never fall back to copying).
- **Python 3.11** (`python3.11` on the path). The orchestrator and the scripts use only the standard library.
- **JDK 25** (Temurin). The engine's mod builds with the included Gradle wrapper; the first build downloads Gradle, Fabric
  Loom and Fabric API, and preparing the world downloads the Fabric server launcher, which fetches the Minecraft 26.3
  server. The Minecraft EULA applies.
- **Memory:** each server takes about 3 GB (`--heap 3G`); the study ran several at once.
- **A model**, for new episodes and judges. The study used `gpt-6.1-sol` (low reasoning effort) and `gpt-6-luna`
  through a ChatGPT plan (provider `chatgpt-plan`). Other providers are `openai-compatible` (`--base-url`, `--model`, a
  key in `OPENAI_API_KEY`), `anthropic` and `mlx-local`; a different model will of course behave differently.

## 1. Build the engine and the town

```sh
export JAVA_HOME=/path/to/jdk-25/Contents/Home
(cd mod && ./gradlew build)             # the mod jar in mod/build/libs, and the Java unit tests
python3.11 scripts/prepare_world.py     # runs/base: the baseline world (downloads the Fabric server launcher once)
python3.11 scripts/build_town.py        # runs/town_jubilee: the riverside town built from scenario/town_jubilee.json
```

`python3.11 scripts/build_town.py --plan-only` writes the town's plan and map without starting a server. Each episode
records the fixture it started from, with the code commit and the mod jar's hash, in its `jubilee_report.json`.

## 2. Replay a shipped episode exactly, with no model ($0)

Every episode in the paper was replayed from its decision log, with no model, and its decisions and world-state hashes
compared with the recorded ones (the queue logs in `runs/queue/*.log` show `PASS replay` for each). To repeat that
check on one episode, copy its shipped logs into the fresh clone's `runs/` (for example by unpacking the logs in another
clone and copying `runs/jub_L2_1`), then:

```sh
python3.11 scripts/run_jubilee.py --runs-dir runs --provider chatgpt-plan --model gpt-6.1-sol --max-calls 0 \
  --name jub_L2_1 --condition scenario/jubilee_conditions/L2_plan.json --port 25600 --game-port 25601 --heap 3G \
  --replay-only
```

It replays into `runs/jub_L2_1_replay` and prints `PASS` or `FAIL`. With `--max-calls 0` no model can be called, so no
sign-in is needed.

## 3. Sign in a model provider

```sh
python3.11 scripts/chatgpt_login.py     # once, for --provider chatgpt-plan; credentials stay in ~/.config/minecraftsim/
```

Every live runner has a hard `--max-calls` budget; a provider's usage limit stops a run the same way, and runners resume
from their latest checkpoint (`--resume`).

## 4. Run episodes

One episode, as run for the paper (Experiment 1, the mayor at motivation L2, a planned surprise):

```sh
python3.11 scripts/run_jubilee.py --runs-dir runs --provider chatgpt-plan --model gpt-6.1-sol --reasoning-effort low \
  --max-calls 3000 --name jub_L2_1 --condition scenario/jubilee_conditions/L2_plan.json --checkpoint-every 4800 \
  --port 25600 --game-port 25601 --heap 3G --replay
```

The condition file sets everything that varies between conditions; `scenario/jubilee_conditions/` has one per
condition: `A_committee` and `L1_plan` to `L3_plan` (planned, by the mayor's motivation), `D_ordinary` and `N3_noplan`
(controls), `E_physics_off`, the exploratory `B_watched` and `C_separate`, Experiment 2's `V2p_plan`, `V2t_told`,
`V2x_physics_off` and `V2o_ordinary`, and `R_fresh` (run with `--ticks 14000`). The episodes in the paper made between
400 and 1,236 model calls each.

**The exact command of every episode, replay and branch in the paper** is in `runs/queue/jobs/`:

```sh
python3 scripts/show_jobs.py              # every job, by kind
python3 scripts/show_jobs.py jub_V2p_3    # one job's command, ports filled in
```

To run many at once, the study used a local queue that runs jobs detached, within a budget of servers, each with its
own ports:

```sh
python3.11 scripts/run_queue.py submit jub_L2_1 --servers 1 -- <command, with {port} and {port+1} for the ports>
python3.11 scripts/run_queue.py serve --max-servers 5 --detach
python3.11 scripts/run_queue.py status
```

The positive controls are scripted ($0): `python3.11 scripts/jubilee_control.py --runs-dir runs --port 25600 --game-port 25601`.

## 5. Judge the episodes

Each judge writes its result into the episode's directory and caches every verdict, so a rerun costs nothing.

```sh
python3.11 scripts/jubilee_leaks.py runs/jub_L2_1 --model gpt-6.1-sol --reasoning-effort low   # what reached the mayor; did it know or suspect
python3.11 scripts/jubilee_investigation.py runs/jub_*[0-9]     # the mayor's requests for news, reports to it, what members noticed
python3.11 scripts/jubilee_mayor_belief.py runs/jub_*[0-9]      # did the mayor believe something was being kept from it
python3.11 scripts/jubilee_commitments.py runs/jub_*[0-9]       # members' commitments against what they did
python3.11 scripts/jubilee_probe.py runs/jub_*[0-9]             # the in-context query at each hour (add --neutral for the neutral question)
python3.11 scripts/jubilee_monitors.py runs/jub_*[0-9] --model gpt-6.1-sol --out runs/jubilee_monitors
python3.11 scripts/jubilee_decomposition.py runs/jub_*[0-9] --model gpt-6-luna
```

Each script's docstring (`python3.11 scripts/<name>.py --help`) describes its inputs and outputs. The judges were
validated against blind labels before use: `scripts/jubilee_judge_validation.py` with the items in
`runs/judge_validation/` (Appendix C).

## 6. Run the branches

A branch restarts a saved episode at one agent's decision, resamples that decision, and runs on with one change
(`scripts/jubilee_fork.py`). The study's branch families each have a submitting script, which queues one job per
episode:

```sh
python3.11 scripts/jubilee_asked.py submit runs/jub_A_[0-9] runs/jub_L3_[0-9] runs/jub_D_[0-9] runs/jub_N3_[0-9] --variant candid
python3.11 scripts/jubilee_asked.py submit runs/jub_V2p_[0-9] runs/jub_V2o_[0-9] --variant counsellor
python3.11 scripts/jubilee_crossings.py submit runs/jub_V2p_[0-9]
```

The noon variants are `asked`, `jubilee`, `candid`, `self`, `drive`, `duty` and `duty_jubilee` (Section 3.5 and
Table 4); each changes one sentence of a brief, quoted in `scripts/jubilee_asked.py`. Judge them with
`scripts/jubilee_asked.py judge runs/jub_*[0-9] --variant <name>`, and the crossings with
`scripts/jubilee_crossings.py report runs/jub_V2p_[0-9]`.

## 7. Analyse

`sh scripts/reproduce.sh` runs the analysis in the paper's order: the branch judgements, Experiment 2's first
suspicions, the crossings, recruitment, the two experiments' analyses, the figures and the report.
