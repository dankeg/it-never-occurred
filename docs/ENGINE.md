# The simulation engine

This is the engine's reference, carried over from its development repository. It describes the harness in general
(including the village scenarios it was first built for); documents it links to there (such as `STATUS.md` or
`docs/MVP_SCOPE_v0.1.md`) are not part of this repository. For this study, start from [`RERUNNING.md`](RERUNNING.md).


A deterministic, lockstep, forkable, physically grounded multi-agent simulation engine on Minecraft Java 26.3 (Fabric).
It is the substrate for *observation-driven simulation* of LLM-agent societies: agents act only through grounded skills,
perceive only what the engine shows them, every run replays exactly from its decision log, and any moment can be forked
into counterfactual branches.

- Scope and gates: `docs/MVP_SCOPE_v0.1.md`. Progress log: `STATUS.md`. Decisions: `DECISIONS.md`.
- Tests, incidents and gates: `docs/TEST_AND_CLEANUP_PLAN.md`. Harness details by milestone: `docs/HARNESS_REFERENCE.md`.

The Minecraft EULA applies. This repository ships only original code (a mod, scripts) and never game files or
decompiled sources.

## Architecture

| Part | What it is |
|---|---|
| `mod/` (Java, Fabric) | The `simharness` server mod: lockstep tick control, server-side agent bodies (bots), deterministic reseeding, state hashing, snapshots and restore, grounded skills, observation and provenance, villagers and their news, communication, documents, the ferry. A JSON-lines control socket on `127.0.0.1` drives it. |
| `scripts/harness.py` | The Python side of the socket (`Client`), one server's lifecycle (`Server`: clones, the jar, ports, a clean stop, `provenance.json`), APFS clones. |
| `orchestrator/` (Python, stdlib only) | The agent runtime: role agents and their memory, prompts, tools, the decision log (every decision with its prompt digest), model clients (Anthropic, OpenAI-compatible, a ChatGPT plan, scripted), the lockstep runtime with replay and forks, the narrated runtime (a game master in place of the engine), scenario predicates. |
| `scripts/village_session.py` | One village run's server session, shared by every village runner. |
| `scripts/` | Fixture builders, runners (village day, studies, forks, narration), gates (`test_*.py`), the gate runner, audits, the local model server, the viewer. `scripts/archive/` holds one-off studies. |
| `scenario/` | The village manifests (roles, places, containers, schematics, predicates, ground truth). |
| `runs/` (not in git) | Fixtures (`base`, `village_<variant>`, `_cache`) and every run directory. Runs are never overwritten. |

## Setup

Requirements: macOS on one APFS volume, Python 3.11 (`/opt/homebrew/bin/python3.11`; the orchestrator and scripts use
only the standard library), Temurin JDK 25 at `~/.local/jdks/jdk-25.0.4.1+1`. The mod builds with the Gradle wrapper
(Gradle 9.7.1, Fabric Loom 1.18-SNAPSHOT, loader 0.19.5, Fabric API 0.162.0+26.3).

```sh
export JAVA_HOME=$HOME/.local/jdks/jdk-25.0.4.1+1/Contents/Home
(cd mod && ./gradlew build)          # the jar (mod/build/libs) and the Java unit tests; --offline once cached
python3.11 scripts/prepare_world.py  # runs/base: the baseline world (downloads the Fabric launcher once)
python3.11 scripts/build_village.py --variant sabotage   # runs/village_sabotage (and --variant accident)
```

A fixture built before fixture metadata existed needs a `fixture.json` before the gate runner accepts it; after
checking it matches its manifest: `python3.11 scripts/provenance.py adopt runs/village_sabotage --manifest
scenario/village_sabotage.json` (and `adopt runs/base` without a manifest).

Tools for tests and lint (not needed to run anything): `pip install pytest ruff` into any Python 3.11 environment.

The local model (optional, about 24 GB of memory) has its own environment, built from the pins in `pyproject.toml`:

```sh
python3.11 -m venv .venv-mlx && .venv-mlx/bin/pip install mlx-lm==0.31.3 mlx==0.32.0 transformers==5.15.0
python3.11 scripts/local_model.py    # serves ~/models/Qwen3.6-35B-A3B-4bit on 127.0.0.1:8080 until interrupted
```

A ChatGPT plan (provider `chatgpt-plan`) needs one sign-in: `python3.11 scripts/chatgpt_login.py`.

### M7c: live bridge incident

After preparing `runs/base`, build each standing-bridge pre-fixture, then stage its incident:

```sh
python3.11 scripts/build_village.py --variant sabotage --pre-incident
python3.11 scripts/stage_incident.py --variant sabotage
python3.11 scripts/build_village.py --variant accident --pre-incident
python3.11 scripts/stage_incident.py --variant accident
```

Pre-fixtures are saved as `runs/village_<variant>_pre`. Staging records the collapse and witness evidence in
`runs/village_<variant>` and preserves an existing fixture as `_prelive`; existing outputs are never overwritten.
Fixture identity comes from provenance's `fixture.json`. IT-09 requires live witness evidence on `staged_live`
fixtures; only older fixtures retain the named `witness_observations` expected failure.

## Workflows

| Task | Command |
|---|---|
| Python unit tests (T1, under 15 s) | `python3.11 -m pytest` |
| Java unit tests (T1j) | `(cd mod && ./gradlew test)` |
| Lint | `ruff check` (rules E and F, line length 120) |
| Gates (servers; see the tiers below) | `python3.11 scripts/run_gates.py --tier fast` (`--list` to see the commands, `--only E3,L1`) |
| A scripted village to the end of day 1 ($0) | `python3.11 scripts/test_m5_village.py --variant sabotage --replay` |
| A live village day (spends calls) | `python3.11 scripts/run_village.py --provider mlx-local --model ~/models/Qwen3.6-35B-A3B-4bit --max-calls 2000 --heap 3G --replay` (or `--provider chatgpt-plan --model <slug>`) |
| A counterfactual study | `python3.11 scripts/run_study.py --study road_closure --branches 10` (`--model scripted` is the $0 default) |
| Fork one logged decision | `python3.11 scripts/fork_decision.py --run <run> --agent <name> --tick <t> --swap '{"name": ..., "args": {...}}'` |
| The narrated village | `python3.11 scripts/run_narrated_village.py --scripted` ($0), or a live GM with `--provider` and `--model` |
| Decision dataset from a run | `python3.11 scripts/build_decision_dataset.py --run <run>` |
| Audits of a run | `scripts/audit_o1.py`, `scripts/audit_testimony.py`, `scripts/audit_grounding.py runs/<run>` |
| Cost and behaviour of a run | `python3.11 scripts/decision_costs.py runs/<run> --rates I,C,O` |
| Offline village metrics ($0) | `python3.11 scripts/village_metrics.py runs/<run>` |
| Watch runs in a browser | `python3.11 scripts/viewer_server.py` (port 8765) |
| What code produced a run | `python3.11 scripts/provenance.py show runs/<run>` |
| Disk retention (plan only) | `python3.11 scripts/prune_runs.py --plan` (see `docs/C7_RETENTION_PLAN.md`) |

Every server-starting script takes `--runs-dir`, `--port` (control socket) and `--game-port`; set both to run beside
another server, and `--heap 3G` when the local model shares the machine.

### Daily life (opt-in)

Legacy village manifests default to `daily_hunger: false`: starting kits, prompts and tools stay unchanged. Build a
separate daily-life fixture with `python3.11 scripts/build_village.py --variant sabotage --daily-hunger`; its fixture
and manifest are named `village_sabotage_daily_life`. Select it in the village runners or dataset builder with
`--variant sabotage_daily_life`. Each role starts with three bread only when enabled.

The harness deducts two food points every 6,000 game ticks (eight per day), reports hunger below six food, and prevents
starvation damage at zero. The enabled agent tool `eat` consumes one held food item after 32 ticks and restores its
vanilla food points and saturation. Food changes and meals produce `hunger` and `eat` events. Engine configuration
can override `hunger_interval_ticks`, `hunger_points` and `eat_ticks`; default values add no legacy saved-state fields.

### Goods and ovens (opt-in)

The Jubilee goods rules are configured together through `config_set`: `carry_limits` maps item identifiers to maximum
inventory totals (default `{}`), `fresh_days` sets bread's lifetime in game days (default `null`), `bake_ticks` sets
ticks per loaf (effective default 40), and `villager_purchases` lists `{stall: [x, y, z], item, count}` market orders
(default `[]`). An explicit `bake_ticks` setting enables the `bake(count)` skill; leaving it absent keeps ovens and
the model-facing baking tool disabled. Disabled settings are omitted from saved configuration so legacy hashes and
observations retain their existing shape. Manifests must include the same enabled config for the model-facing tools.

Every agent gain, including automatic pickup, must fit its full item load; overweight requests fail with `too_heavy`
and a capacity hint instead of silently taking less. Bread stores `simharness_baked_day` in `minecraft:custom_data`;
unstamped bread was baked on day zero, and `today - baked_day > fresh_days` makes it mouldy and inedible. Inventories
and opened containers show each stack's baking day or mould. Baking needs a visible smoker in reach and three wheat
per loaf, with 1–32 loaves per batch. Its smoker glows throughout the saved timer and unlights on success, failure or
cancellation; only successful completion consumes wheat and delivers day-stamped bread.

With any goods rule enabled, a shop's `acl_set` owner may stock its chest and take takings (the ACL still applies);
other agents must use `buy`. Each scripted adult's actual market arrival triggers configured orders, freshest
non-mouldy stock first, paying the posted emerald price per item into the chest. Purchases are ordered by arrival tick
and villager ID and log ordinary `purchase` events; unavailable stock logs `sold_out` without a partial sale.
`scripts/test_it23_goods.py` is the isolated server probe for these rules, including default-off hashes and checkpoint
replay; run it only on a clone of `runs/village_sabotage`, after building the mod.

### Offline village metrics

`scripts/village_metrics.py` writes and prints `<run>/metrics.json` without starting a server or changing a run's
decisions. Village reports and finished study outcomes also write it automatically. It reports daily commute and
attendance summaries, purchase/transfer counts and quantities, posted prices, role emerald holdings, treasury and
Gini, quoted promises checked against logged delivery, the bridge contract's payment outcome, and investigation
scores from saved judgments or the existing keyword judge.

The keyword extractor is free. To opt into model-based extraction, use
`python3.11 scripts/village_metrics.py runs/<run> --extractor llm --provider openai-compatible --model MODEL --base-url URL`.
It makes one call per speaker/day, so this option can spend provider credits. Unsupported quotes are dropped and
counted; this never changes agent prompts or tools.

Days are half-open windows `[day*24000, (day+1)*24000)`; partial days stop at the report's end tick. Holdings are each
role's emeralds as its latest prompt showed them (its starting kit before its first decision), plus the logged
transfers and purchases since; the Gini covers the six roles. The treasury is separate from Gini. Prices retain the latest parseable stall-sign edit;
malformed edits are reported, not interpreted. Activity shares use scheduled villagers; late school arrivals count
as reaching school. Missing attendance denominators are `null`. World completion times reuse tracked first-true
ticks or successful construction-skill events, not a new block scan. Keyword investigation scores measure what a
report says, not whether its evidence is independently grounded.

### Public-evidence auditors

`scripts/auditor.py` asks a question from a supplied JSON file using only
`RunViews.render()`'s public speech/actions, never private notes, prompts or goals. Questions are a JSON array;
each entry contains `id`, `system`, `question` and an object JSON-schema `verdict`. No research questions ship in
code. The dependency-free validator supports nested properties, required/extra fields, arrays, scalar/nullable
types, enums/constants, numeric bounds, lengths, patterns and schema composition; unsupported keywords (including
references) are rejected before a call rather than silently ignored.

```bash
python3.11 scripts/auditor.py --questions questions.json --question QUESTION_ID --view both \
  --provider chatgpt-plan --model MODEL --reasoning-effort low --until 12000 \
  --out results.jsonl runs/RUN
python3.11 scripts/auditor_scores.py results.jsonl labels.json --positive POSITIVE_LABEL --seed 0
```

Auditing can spend provider credits. It reuses `run_village.model_client` (fixed temperature 0 where applicable),
requires one `verdict` tool call and retries malformed answers once. Each actual client response, including a failed
attempt, is recorded with purpose `auditor` in `runs/ledger.jsonl`. Valid answers are cached under
`runs/auditor_cache/<sha256>.json`; the hash covers canonical UTF-8 JSON of the model, reasoning effort, complete
question entry and rendered view. A cache hit spends nothing. Evidence longer than `--max-chars` (default 400,000)
is refused even on a cache hit; use `--until` or a narrower view instead of truncating it.

Results append one JSONL row per run with `run`, `question`, `view`, `model`, `view_chars`, `request_digest` and a
nested `verdict`. For scoring, verdicts need `label` and `confidence`; confidence is the score/probability for the
class named by `--positive`, not confidence in whichever label was selected. Labels are a JSON object mapping run
names to true labels. Scores group by question/view/model and report sample count, tie-aware AUROC, label accuracy,
FPR at target TPR (`--tpr`, default 0.9) and seeded 95% run-bootstrap intervals. The operating point is the first
descending-score threshold reaching the target TPR, accepting ties together. Missing labels are listed and excluded;
metrics requiring both classes are `null` when undefined. Class-degenerate bootstrap replicates are omitted only
for those undefined metrics. `--bootstrap` sets the replicate count (default 2,000); duplicate runs within one
question/view/model group are rejected rather than counted as independent evidence.

## Test tiers

| Tier | Command | Budget | Contents |
|---|---|---|---|
| T1 | `python3.11 -m pytest` | 15 s | unit tests, golden prompts (`scripts/tests/golden/`), regression tests named after their fixes |
| T1j | `./gradlew test` (in `mod/`) | 2 min | Java unit tests |
| T3 fast | `run_gates.py --tier fast` | 5 min | E3, L1, N1 plumbing and the fast integration tests |
| T4 full | `run_gates.py --tier full` | 60 min | fast, E1, E2, E2-skills, O1, S1, M5 + V-G2 (both variants), the scripted narrated village, the replay oracle, the slower integration tests, the probes |
| T5 model | `run_gates.py --tier model` | 10 min | the local model (needs the machine to itself) |
| weekly | `run_gates.py --tier weekly` | hours | V1 (29 min), fixtures from a clean checkout, the soak |

The gate runner refuses to start if free memory or disk is short, or a fixture is missing, unrecorded or stale. It
records the commit, a dirty tree's diff and the jar's SHA-256, stops if the jar changes mid-chain, gives every gate
fresh fixture clones and its own ports, and writes `runs/gates_<date>_<n>/gates.json` after every gate, with triage
(first divergence, server log tail, the exact command) for each failure. Which tiers a change must pass before merge is
the change-to-test map in the plan (§7).

## Run policy

- **Exact replay is the oracle.** A run replays from its `decisions.jsonl` with no model only while prompts are
  byte-stable; the golden prompt tests guard that. An intentional prompt change updates the golden files
  (`python3.11 scripts/tests/test_golden_prompts.py --update`) and needs a fresh recorded oracle run.
- **Never rebuild the jar while servers from this checkout are running**: a server installs the jar at start, so a
  chain would run two builds (incident P1).
- **Never overwrite or delete runs.** Scripts refuse existing run directories; studies set unfinished branches aside.
  Pruning is planned (`prune_runs.py --plan`) and approved by the owner first.
- **Memory:** the local model takes about 24 GB of a 32 GB machine; run servers beside it only with `--heap 3G`, one
  chain at a time, and not at all while another agent's model is loaded.
- **Spending:** live model calls need approval; every live runner has a hard `--max-calls` budget, and a provider's
  usage limit stops a run like a budget.
- **Provenance:** every server start appends the commit, the dirty flag (diff saved beside the run) and the jar hash to
  the run's `provenance.json`; record runs from a clean, committed tree.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ReplayDiverged` and a `diverged.json` | A prompt differs from the recorded one: code changed since the run was recorded (see its `provenance.json`). Replay with the recorded commit's code, or hold agents to their logged actions (`fork_decision.py --forced`, `build_decision_dataset.py --forced` or `--prompt-context`). |
| A server will not start: address in use | Another server holds the control or game port; pass other `--port`/`--game-port`. A leftover JVM from an interrupted run: long-running scripts stop their servers on SIGINT and SIGTERM (C8); check `lsof -t runs/<run>/server.log`. |
| The gate runner refuses: insufficient memory | A local model or other servers are running; wait, or run fewer gates. |
| The gate runner refuses: no `fixture.json` or stale | Rebuild the fixture, or adopt it after checking (`scripts/provenance.py adopt`). |
| `APFS clone failed` or `not a verified APFS volume` | `runs/` must be on the same APFS volume as its fixtures; clones never fall back to copying. |
| `Build the harness mod first` | `(cd mod && ./gradlew build)`. The first build after a dependency change needs the network once (drop `--offline`). |
| A run stopped with `BudgetExceeded` | The `--max-calls` budget or the provider's usage limit; the run's decisions so far are in its log, and studies resume from finished branches. |
| `Not signed in: run scripts/chatgpt_login.py` | Sign in once; credentials are kept owner-only in `~/.config/minecraftsim/chatgpt.json`. |
