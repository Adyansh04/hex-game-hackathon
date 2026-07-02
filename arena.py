#!/usr/bin/env python
"""
Local Hex arena for testing agents head-to-head.

Plays an agent A against an agent B over a set of (optionally random) openings,
running each opening once with A as Player 0 and once with A as Player 1, so the
comparison is color-balanced. Reports win rates by color and per-move timing.

Agents run IN-PROCESS (like gamelib-play). Per-move time budgets are scaled by
monkeypatching known class attributes (SOFT/HARD_TIME_LIMIT_SECONDS,
MOVE_TIME_LIMIT_SECONDS) so you can run fast iteration games at e.g. --time 0.5
and final validation at --time 4.5. ch5/final_agent is never edited on disk.

Usage examples:
  python arena.py final_agent.py final_agent.py --games 1 --openings 0 --time 0.5
  python arena.py v1_agent.py final_agent.py --games 8 --openings 2 --time 0.6 --jobs 6
  python arena.py v1_agent.py final_agent.py --games 4 --openings 2 --time 4.5 --cap 5.0 --jobs 4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = os.path.dirname(os.path.abspath(__file__))

TIME_ATTRS_SOFT = ["SOFT_TIME_LIMIT_SECONDS"]
TIME_ATTRS_HARD = ["HARD_TIME_LIMIT_SECONDS", "MOVE_TIME_LIMIT_SECONDS"]


def load_agent_class(path):
    """Import a .py file by path and return its gamelib Agent subclass."""
    path = os.path.abspath(path)
    base = os.path.splitext(os.path.basename(path))[0]
    modname = f"agent_mod_{base}_{abs(hash(path)) % 100000}"
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    from gamelib.hex.agent import Agent

    found = None
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and issubclass(obj, Agent) and obj is not Agent:
            found = obj
    if found is None:
        raise RuntimeError(f"No gamelib Agent subclass found in {path}")
    return found


def set_time_budget(cls, seconds):
    """Scale an agent class's self-imposed time limits to ~`seconds`."""
    if seconds is None:
        return
    for a in TIME_ATTRS_SOFT:
        if hasattr(cls, a):
            setattr(cls, a, max(0.05, seconds * 0.92))
    for a in TIME_ATTRS_HARD:
        if hasattr(cls, a):
            setattr(cls, a, max(0.06, seconds))


def random_opening(board_size, num_plies, seed):
    """Return a list of (r, c) cells for a random alternating opening."""
    rng = random.Random(seed)
    cells = [(r, c) for r in range(board_size) for c in range(board_size)]
    rng.shuffle(cells)
    return cells[:num_plies]


def _mk_state(board_size, opening):
    from gamelib.hex.gamestate import GameState as State
    from gamelib.hex.engine import Engine
    from gamelib.hex.move import Move

    engine = Engine()
    state = State.initial({"board_size": board_size})
    for (r, c) in opening:
        p = state.turn
        mv = Move(player=p, position=[r, c])
        if not engine.validate_move(state, mv):
            break
        state = engine.apply_move(state, mv)
    return engine, state


def play_game(pathP0, pathP1, opening, board_size, time_budget, cap):
    """Play one game. Returns a result dict. Loser is decided on illegal/timeout/exception."""
    from gamelib.hex.engine import Engine  # noqa: F401 (import cost warms per-worker)

    ClsP0 = load_agent_class(pathP0)
    ClsP1 = load_agent_class(pathP1)
    set_time_budget(ClsP0, time_budget)
    set_time_budget(ClsP1, time_budget)

    engine, state = _mk_state(board_size, opening)
    if engine.is_game_over(state):
        return {"winner": state.status, "reason": "opening_decided", "plies": len(opening),
                "max_t": [0.0, 0.0], "over_cap": False}

    a0 = ClsP0(); a0.initialize({"player_id": 0})
    a1 = ClsP1(); a1.initialize({"player_id": 1})
    agents = [a0, a1]

    max_t = [0.0, 0.0]
    over_cap = False
    plies = len(opening)
    while not engine.is_game_over(state):
        p = int(state.turn)
        t0 = time.perf_counter()
        try:
            mv = agents[p].get_move(state)
        except Exception as e:  # noqa: BLE001
            return {"winner": 1 - p, "reason": f"exception_p{p}", "err": repr(e),
                    "trace": traceback.format_exc()[-1500:], "plies": plies, "max_t": max_t,
                    "over_cap": over_cap}
        dt = time.perf_counter() - t0
        if dt > max_t[p]:
            max_t[p] = dt
        if cap is not None and dt > cap:
            over_cap = True
            return {"winner": 1 - p, "reason": f"timeout_p{p}({dt:.2f}s)", "plies": plies,
                    "max_t": max_t, "over_cap": True}
        if not engine.validate_move(state, mv):
            return {"winner": 1 - p, "reason": f"illegal_p{p}({mv.position})", "plies": plies,
                    "max_t": max_t, "over_cap": over_cap}
        state = engine.apply_move(state, mv)
        plies += 1

    return {"winner": int(state.status), "reason": "connection", "plies": plies,
            "max_t": max_t, "over_cap": over_cap}


def _worker(job):
    """Top-level worker for multiprocessing. job = dict of args."""
    res = play_game(job["p0"], job["p1"], job["opening"], job["board"], job["time"], job["cap"])
    res["job"] = {k: job[k] for k in ("idx", "orientation", "opening")}
    return res


def main():
    ap = argparse.ArgumentParser(description="Local Hex arena")
    ap.add_argument("agentA")
    ap.add_argument("agentB")
    ap.add_argument("--games", type=int, default=6, help="number of distinct openings")
    ap.add_argument("--openings", type=int, default=2, help="random opening plies per game (0 = agents' own opening)")
    ap.add_argument("--time", type=float, default=0.6, help="per-move time budget scaled into each agent")
    ap.add_argument("--cap", type=float, default=None, help="if a move exceeds this many seconds, it's a loss (e.g. 5.0)")
    ap.add_argument("--board", type=int, default=11)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=str, default=None, help="write JSON results here")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    pathA = os.path.abspath(args.agentA)
    pathB = os.path.abspath(args.agentB)
    nameA = os.path.basename(pathA)
    nameB = os.path.basename(pathB)

    # Build jobs: each opening played twice (A as P0, then A as P1).
    jobs = []
    for i in range(args.games):
        opening = [] if args.openings <= 0 else random_opening(args.board, args.openings, args.seed + i)
        # openings==0 => only 1 distinct game per orientation; avoid duplicates.
        jobs.append({"idx": i, "orientation": "A_as_P0", "p0": pathA, "p1": pathB,
                     "opening": opening, "board": args.board, "time": args.time, "cap": args.cap})
        jobs.append({"idx": i, "orientation": "A_as_P1", "p0": pathB, "p1": pathA,
                     "opening": opening, "board": args.board, "time": args.time, "cap": args.cap})
        if args.openings <= 0:
            break  # openings==0 => all games identical; one pair is enough

    print(f"Arena: A={nameA}  vs  B={nameB}")
    print(f"  games={args.games} openings={args.openings} time={args.time}s cap={args.cap} "
          f"board={args.board} jobs={args.jobs}  ({len(jobs)} total games)")
    print("-" * 72)

    results = []
    t_start = time.perf_counter()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_worker, j): j for j in jobs}
            for fut in as_completed(futs):
                results.append(fut.result())
                _print_line(results[-1], nameA, nameB, args.verbose)
    else:
        for j in jobs:
            results.append(_worker(j))
            _print_line(results[-1], nameA, nameB, args.verbose)

    # Tally from A's perspective.
    a_p0_games = a_p0_wins = 0
    a_p1_games = a_p1_wins = 0
    max_move_time = 0.0
    over_cap_count = 0
    for r in results:
        orient = r["job"]["orientation"]
        winner = r["winner"]  # 0 or 1
        max_move_time = max(max_move_time, max(r["max_t"]))
        if r.get("over_cap"):
            over_cap_count += 1
        if orient == "A_as_P0":
            a_p0_games += 1
            if winner == 0:
                a_p0_wins += 1
        else:  # A_as_P1
            a_p1_games += 1
            if winner == 1:
                a_p1_wins += 1

    total_games = a_p0_games + a_p1_games
    total_wins = a_p0_wins + a_p1_wins
    elapsed = time.perf_counter() - t_start
    print("-" * 72)
    print(f"A = {nameA}")
    print(f"  as P0 : {a_p0_wins}/{a_p0_games} wins")
    print(f"  as P1 : {a_p1_wins}/{a_p1_games} wins")
    print(f"  TOTAL : {total_wins}/{total_games} wins  ({100.0*total_wins/max(1,total_games):.1f}%)")
    print(f"  max single move time across all games: {max_move_time:.2f}s   over-cap games: {over_cap_count}")
    print(f"  wall time: {elapsed:.1f}s")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"A": nameA, "B": nameB, "args": vars(args), "results": results,
                       "summary": {"a_p0": [a_p0_wins, a_p0_games], "a_p1": [a_p1_wins, a_p1_games],
                                   "total": [total_wins, total_games], "max_move_time": max_move_time,
                                   "over_cap": over_cap_count}}, f, indent=2)
        print(f"  wrote {args.out}")


def _print_line(r, nameA, nameB, verbose):
    orient = r["job"]["orientation"]
    who = {0: "P0", 1: "P1"}[r["winner"]]
    winner_name = nameA if ((orient == "A_as_P0" and r["winner"] == 0) or (orient == "A_as_P1" and r["winner"] == 1)) else nameB
    tag = "A" if winner_name == nameA else "B"
    line = f"  game {r['job']['idx']:>2} {orient:<8} -> {who} wins [{tag}={winner_name}]  ({r['reason']}, {r['plies']} plies, maxt={max(r['max_t']):.2f}s)"
    print(line)
    if verbose and "trace" in r:
        print(r["trace"])


if __name__ == "__main__":
    main()
