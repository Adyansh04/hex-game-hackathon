# Hex agent visualizer

An interactive "x-ray" of what **`v7_resistance_agent.py`** computes during a real
game. It plays the champion against an opponent, records every decision, and produces a
**single self-contained HTML file** you open in a browser — no server, no install.

![how it works](../final_match.gif)

## What it shows (step through the game with the scrubber / ← → keys)

For each of v7's moves:

- **Electrical resistance** — the board as a resistor network: per-cell **potentials**
  (heatmap) and **current flow** along edges (thicker = more current), for either
  player. This is literally "which connections carry the signal", plus `R` for each side
  and the eval score `1000·ln(R_opp/R_me)`.
- **Alpha-beta search tree** — the explored tree: each node shows the move, depth,
  `[α,β]` window and returned score, with badges for **cutoffs (✂N pruned)**,
  **transposition-table hits (TT)**, **leaf** evals, **WIN/LOSS**, and PVS **probe**
  (null-window) nodes. The **principal variation** is highlighted. Collapsible; click a
  node to flash its move on the board.
- **Shortest paths & ranking** — both players' Dijkstra connection paths, and the ranked
  root candidates with a full **score-component breakdown** (race, disruption, static,
  save-bridge, path bonuses, edge penalty, danger).
- **Bridges & tactics** — formed bridges + carriers, save-bridge replies, and
  immediate-win / block markers.
- **Per-move stats** — decision provenance (opening book / immediate win / block /
  search), nodes, TT size, time, and the **iterative-deepening progression** table.
- **Game-level charts** — evaluation, `ln(resistance)` for both sides, nodes and time,
  across the whole game, with a cursor on the current move.

Toggle any overlay on/off; pick which player's resistance to display.

## Generate a visualization

Use the `hex` conda env (Python 3.12 + numpy). From the repo root:

```bash
python visualizer/record_game.py --opponent v6_hackathon_agent.py --time 0.5 --depth 3 --out visualizer/output/demo
```

Then open **`visualizer/output/demo.html`** in any browser (double-click it).

Flags:

| flag | default | meaning |
|---|---|---|
| `--opponent` | `v6_hackathon_agent.py` | any agent `.py` in the repo |
| `--side` | `0` | which side v7 plays (0 = Left↔Right, 1 = Top↔Bottom) |
| `--time` | `0.5` | per-move budget (seconds) |
| `--depth` | `3` | max search depth — **kept small so the tree stays legible** |
| `--openings` | `0` | random opening plies (0 = agents' own openings) |
| `--out` | `visualizer/output/demo` | output base path (`.json` + `.html`) |

> The recorded game runs at a **reduced search budget** so the alpha-beta tree has a few
> hundred readable nodes instead of ~10,000. The resistance / path / ranking analyses
> are computed at full fidelity for each shown position. Raise `--depth`/`--time` for a
> stronger (but bushier) tree.

## How it works

- `tracer.py` — `TracingHexAgent`, a **subclass** of the champion. It plays exactly like
  `v7` but records the analysis by wrapping the real `_alpha_beta` / `_resistance` /
  ranking methods via `super()` (so the trace can never drift from the real logic). The
  champion file is **not modified**.
- `record_game.py` — drives a full game with the gamelib engine (reusing `arena.py`'s
  agent loader) and inlines the renderer assets + trace into one standalone HTML.
- `assets/` — the zero-dependency renderer (`viewer.html`, `viz.css`, `viz.js`): SVG
  board, overlays, collapsible tree, charts, controls.
- `output/` — generated `.json` + `.html` (git-ignored; `git add -f` a sample to ship one).
