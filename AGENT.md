# AGENT.md — Operating Manual for Coding Agents

## Project

**Topic:** Development of a Micro Language Model for Defence, Aerospace & Public
Enterprises (HAL, ISRO, DRDO) using the TensorFlow Framework:
**BHEL-OptimizeTensorFlow Agents for Autonomous Boiler Lifecycle Monitoring in
Thermal Power Plants.**

In plain terms: a small (micro) language model built with TensorFlow that powers
autonomous monitoring agents for boiler lifecycle management (start-up, steady
operation, anomalies, degradation, maintenance, shutdown) in thermal power
plants, exposed through a Flask web UI.

## Environment (NixOS)

- This machine runs **NixOS** inside a nix-flake that auto-enables in
  `~/projects` and subdirectories.
- **Never** use `apt`, `dnf`, or any other system package manager.
- **Never** use nix commands to work around a missing package. If a needed
  package/tool is not present in the flake, **stop and inform the user**, then
  wait for their decision.
- Python tooling must go through **`uv`** (already available). Prefer
  `uv add <pkg>` over `uv pip install`. Use `uv run` to execute project code.

## Tech Stack (fixed unless the user says otherwise)

- **Language:** Python 3.13 (managed by `uv`)
- **ML framework:** TensorFlow 2.21 (Keras API) — **GPU build, verified working**
  (`GPU:0` = RTX 4050 Laptop; nix-ld enabled so the pip wheel loads on NixOS)
- **Backend/UI:** Flask (Jinja templates + minimal CSS/JS for dashboards)
- **Data:** CSV/JSON synthetic + reference datasets (no proprietary data)
- **Testing:** `pytest`
- **Style:** PEP 8, type hints where practical, docstrings on public functions
- **Package:** import as `bhel_optimize` (uv project name `bhel_optimize`)
- **Lockfile:** `uv.lock` is committed — keep it in sync with `uv add`/`uv sync`

## Repository Layout (evolve as needed)

```
MPRVII/
├── AGENT.md          # this file — rules for agents
├── ROADMAP.md        # the bible: all tasks, checkpoints, status
├── plan.md           # approved Phase 1 plan + future phases
├── .agents/          # onboarding knowledge & working notes (see below)
│   ├── boiler.md     #   boiler anatomy, sensors, failure modes
│   ├── dataset.md    #   dataset facts, all 30 tags, labelling rule
│   └── architecture.md  # system architecture (created in implementation)
├── pyproject.toml    # uv-managed project (package: bhel_optimize)
├── uv.lock           # committed lockfile (reproducible TF/CUDA pins)
├── data/             # datasets (git-ignored, user-supplied CSVs)
├── artifacts/        # generated: scalers, models, leaderboards (git-ignored)
├── src/bhel_optimize/
│   ├── config.py     #   tags, part map, criticality, thresholds (source of truth)
│   ├── data/         #   loader.py, preprocess.py
│   ├── simulation/   #   scenario controller + fault injection (M3)
│   ├── detection/    #   per-part hybrid ensembles (M4)
│   ├── overview/     #   Overview Agent (M5)
│   └── server/       #   Flask app (M6)
├── templates/        # Flask templates
├── static/           # CSS/JS assets
└── tests/            # pytest suites
```

## Onboarding Knowledge (`.agents/`) — READ FIRST

New agent sessions **must** read these before writing code; they are the
project's anti-context-drift memory:

- **`.agents/boiler.md`** — boiler anatomy, monitored parts/sensors, creeping
  and structural failure modes. **Append/update it** whenever new domain
  knowledge is learned.
- **`.agents/dataset.md`** — dataset facts, all 30 tags, the anomaly-labelling
  rule, dataset quirks. **Append/update it** when data findings change
  (e.g. measured vs. paper-stated values).
- **`.agents/architecture.md`** — created during implementation; keep the data
  → detection → overview → UI flow and module map up to date.

`.agents/` files record **verified facts only**. Never record a dataset, tool
or paper as fact unless it was checked in this project. If something is
unverified, say so explicitly or omit it.

## Workflow Rules (non-negotiable)

0. **Never jump ahead of the user.** Do not begin a new task, file, or
   component without the user's explicit go-ahead. When confused or facing
   ambiguity, **ask questions** (use the AskUserQuestion tool) instead of
   guessing. After completing what was asked, **wait for further instructions**
   before proceeding. Do not pre-emptively plan or build things the user has
   not described yet (e.g., do not draft ROADMAP.md content until the user
   has explained the plan).

1. **ROADMAP.md is the bible.** Before starting work, read it. While working,
   keep only the current task in mind. When a task is done, immediately mark its
   checkbox `[x]` in ROADMAP.md — never batch updates.
2. **Roadmap growth:** append new checkpoints, tasks, and subtasks to
   ROADMAP.md as the project reveals them. Never delete completed items — they
   are the audit trail. Mark blocked items with `⚠ blocked:` and a reason.
3. **Commit discipline:**
   - Git commits happen whenever a task **or a coherent group of subtasks** is
     complete — i.e., at every major-task boundary ("about to begin a new major
     task").
   - Commit messages: **≤ 15 words**, concise, clear, imperative mood
     (e.g., "Add boiler sensor data generator and unit tests").
   - Never stack 20 tasks into one commit. If several small related items finish
     together (one logical unit), one commit is fine; unrelated items get their
     own commits.
4. **Git:** the project has its own repo (nested in the parent flake repo).
   Commit the AGENT.md/ROADMAP.md skeleton first, then code as it lands.
5. **Frontend:** Flask server only. Keep it basic — server-rendered templates,
   small vanilla JS for live charts if needed. No heavy SPA frameworks.
6. **Agents must verify their work:** run tests / start the Flask app before
   declaring a task complete.
7. **Notes:** put decisions, research summaries, and design sketches in
   `.agents/` — and keep the onboarding docs (`boiler.md`, `dataset.md`,
   `architecture.md`) updated as the project learns. Keep ROADMAP.md clean and
   task-focused.

## Definition of Done (per task)

- Code written and committed.
- Tests pass (`uv run pytest`) or, for UI tasks, the Flask app renders the new
  page without errors.
- ROADMAP.md checkbox ticked.
- Commit made with a ≤15-word message.
