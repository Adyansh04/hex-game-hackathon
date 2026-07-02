"""
Record an instrumented game and emit a self-contained interactive visualizer.

Plays `v7_resistance_agent.py` (as a TracingHexAgent) against an opponent, captures a
full per-move analysis trace, and writes:
  - output/<name>.json   the raw trace
  - output/<name>.html   a standalone page (assets + data inlined; just open it)

Usage (from the repo root, using the `hex` conda env):
  python visualizer/record_game.py --opponent v6_hackathon_agent.py --time 0.5 --depth 3 --out visualizer/output/demo

Reduced --depth/--time keep the alpha-beta tree legible; resistance / path / ranking
analyses are full fidelity regardless.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_ASSETS = os.path.join(_HERE, "assets")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import arena  # reuse the agent loader + time-budget helpers
import tracer  # the instrumented agent (also loads the champion)

from gamelib.hex.gamestate import GameState as State
from gamelib.hex.engine import Engine
from gamelib.hex.move import Move


def build_trace(opponent_path, side, budget_time, depth, openings, seed):
    n = 11
    OppCls = arena.load_agent_class(opponent_path)
    T = tracer.TracingHexAgent

    # Reduced budget: cap depth for a legible tree; time as a safety net.
    T.SOFT_TIME_LIMIT_SECONDS = max(0.05, budget_time * 0.92)
    T.HARD_TIME_LIMIT_SECONDS = max(0.06, budget_time)
    T.MAX_SEARCH_DEPTH = depth
    arena.set_time_budget(OppCls, budget_time)

    engine = Engine()
    state = State.initial({"board_size": n})

    opening = [] if openings <= 0 else arena.random_opening(n, openings, seed)
    history = []
    for (r, c) in opening:
        p = int(state.turn)
        mv = Move(player=p, position=[r, c])
        if not engine.validate_move(state, mv):
            break
        history.append({"ply": len(history), "player": p, "cell": r * n + c, "opening": True})
        state = engine.apply_move(state, mv)

    v7 = T()
    v7.initialize({"player_id": side})
    opp = OppCls()
    opp.initialize({"player_id": 1 - side})
    agents = {side: v7, 1 - side: opp}

    reason = "connection"
    max_plies = n * n + 2
    while not engine.is_game_over(state) and len(history) < max_plies:
        p = int(state.turn)
        ag = agents[p]
        try:
            mv = ag.get_move(state)
        except Exception as exc:  # noqa: BLE001
            reason = f"exception_p{p}: {exc!r}"
            break
        if not engine.validate_move(state, mv):
            reason = f"illegal_p{p}: {mv.position}"
            break
        cell = int(mv.position[0]) * n + int(mv.position[1])
        history.append({"ply": len(history), "player": p, "cell": cell})
        if p == side:
            v7.traces[-1]["ply"] = len(history) - 1
            v7.traces[-1]["move_number"] = sum(1 for h in history if h["player"] == side)
        state = engine.apply_move(state, mv)

    winner = int(state.status) if state.status in (0, 1) else None
    v7_name = "v7_resistance_agent.py"
    opp_name = os.path.basename(opponent_path)
    names = {side: v7_name, 1 - side: opp_name}
    trace = {
        "meta": {
            "board_size": n,
            "v7": v7_name,
            "v7_side": side,
            "opponent": opp_name,
            "budget": {"time": budget_time, "depth": depth},
            "openings": openings,
            "winner": winner,
            "winner_name": (names.get(winner) if winner is not None else None),
            "v7_won": (winner == side),
            "plies": len(history),
            "reason": reason,
        },
        "history": history,
        "moves": v7.traces,
    }
    return trace


def emit_html(trace, out_base):
    with open(os.path.join(_ASSETS, "viewer.html"), encoding="utf-8") as f:
        tpl = f.read()
    with open(os.path.join(_ASSETS, "viz.css"), encoding="utf-8") as f:
        css = f.read()
    with open(os.path.join(_ASSETS, "viz.js"), encoding="utf-8") as f:
        js = f.read()
    data = json.dumps(trace, separators=(",", ":"))
    html = (tpl.replace("/*__CSS__*/", css)
               .replace("HEXVIZ_DATA_PLACEHOLDER", data)
               .replace("/*__JS__*/", js))
    html_path = out_base + ".html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def main():
    ap = argparse.ArgumentParser(description="Record an instrumented Hex game and build a visualizer.")
    ap.add_argument("--opponent", default="v6_hackathon_agent.py",
                    help="opponent agent .py (relative to repo root or absolute)")
    ap.add_argument("--side", type=int, default=0, choices=[0, 1], help="which side v7 plays")
    ap.add_argument("--time", type=float, default=0.5, help="per-move time budget (s)")
    ap.add_argument("--depth", type=int, default=3, help="max search depth (keeps the tree legible)")
    ap.add_argument("--openings", type=int, default=0, help="random opening plies (0 = agents' own openings)")
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--out", default=os.path.join("visualizer", "output", "demo"),
                    help="output base path (…json / …html)")
    args = ap.parse_args()

    opp_path = args.opponent
    if not os.path.isabs(opp_path):
        opp_path = os.path.join(_REPO, opp_path)

    out_base = args.out
    if not os.path.isabs(out_base):
        out_base = os.path.join(_REPO, out_base)
    os.makedirs(os.path.dirname(out_base), exist_ok=True)

    t0 = time.perf_counter()
    print(f"Recording: v7 (side {args.side}) vs {os.path.basename(opp_path)} "
          f"| depth={args.depth} time={args.time}s openings={args.openings}")
    trace = build_trace(opp_path, args.side, args.time, args.depth, args.openings, args.seed)

    with open(out_base + ".json", "w", encoding="utf-8") as f:
        json.dump(trace, f, separators=(",", ":"))
    html_path = emit_html(trace, out_base)

    m = trace["meta"]
    result = ("v7 WON" if m["v7_won"] else ("draw/none" if m["winner"] is None else "v7 lost"))
    print(f"Done in {time.perf_counter()-t0:.1f}s | {m['plies']} plies | {result} "
          f"| v7 moves recorded: {len(trace['moves'])}")
    print(f"  JSON: {out_base + '.json'}")
    print(f"  HTML: {html_path}   (open in a browser)")


if __name__ == "__main__":
    main()
