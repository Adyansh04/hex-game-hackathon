"""
TracingHexAgent — an instrumented subclass of the champion agent.

It plays exactly like `v7_resistance_agent.py` (it IS that agent, subclassed) but
records, for every one of its own `get_move` decisions, a JSON-serializable trace of
the internal analysis: the electrical-resistance networks (potentials + currents) for
both players, Dijkstra shortest paths, the ranked root candidates with score
components, bridges + tactical markers, the iterative-deepening progression, and the
alpha-beta search tree (recorded by wrapping the real search via super(), so it can
never drift from the agent's actual logic).

The champion file itself is NOT modified. Run through `record_game.py`.
"""
import os

# Match the champion's BLAS setup BEFORE numpy is imported anywhere.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import importlib.util
import math
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHAMP_PATH = os.path.join(_REPO, "v7_resistance_agent.py")

_spec = importlib.util.spec_from_file_location("v7_champion", _CHAMP_PATH)
champ = importlib.util.module_from_spec(_spec)
sys.modules["v7_champion"] = champ
_spec.loader.exec_module(champ)

import numpy as np  # noqa: E402  (champion already imported numpy with 1-thread BLAS)

HexAgent = champ.HexAgent
EMPTY = champ.EMPTY
PLAYER_0 = champ.PLAYER_0
PLAYER_1 = champ.PLAYER_1

_INFCAP = 1e8
_WIN = 900_000.0  # treat |score| above this as terminal win/loss for display


def _r(x):
    """Round for JSON; clamp infinities to a sentinel the viewer understands."""
    if x is None:
        return None
    x = float(x)
    if x >= _INFCAP:
        return _INFCAP
    if x <= -_INFCAP:
        return -_INFCAP
    return round(x, 2)


class TracingHexAgent(HexAgent):
    """Champion agent that records a per-move analysis trace into self.traces."""

    # Cap on recorded search nodes per move (protects JSON size; search itself is
    # unaffected — we simply stop *recording* deeper once the cap is hit).
    NODE_RECORD_BUDGET = 6000

    def initialize(self, init_data):
        super().initialize(init_data)
        self.traces = []
        self._rec = None
        self._node_stack = []          # [(node_dict, board_snapshot_list), ...]
        self._depth_trees = {}         # depth -> root node for that ID iteration
        self._last_completed_depth = 0
        self._nodes_recorded = 0
        self._move_t0 = 0.0

    # ------------------------------------------------------------------
    # get_move: set up a fresh recorder, run the real pipeline, finalize.
    # ------------------------------------------------------------------
    def get_move(self, state):
        self._rec = {"stage": None, "id": []}
        self._depth_trees = {}
        self._last_completed_depth = 0
        self._nodes_recorded = 0
        self._node_stack = []
        self._move_t0 = time.perf_counter()
        move = super().get_move(state)
        try:
            self._finalize_trace(state, move)
        except Exception as exc:  # never let tracing break the game
            self._rec["trace_error"] = repr(exc)
        self.traces.append(self._rec)
        self._rec = None
        return move

    # ------------------------------------------------------------------
    # Provenance hooks (which pipeline stage decided the move)
    # ------------------------------------------------------------------
    def _opening_move(self, board, player, legal_moves):
        res = super()._opening_move(board, player, legal_moves)
        if self._rec is not None and res is not None and self._rec.get("stage") is None:
            self._rec["stage"] = "opening_book"
        return res

    def _find_immediate_win(self, board, hash_key, bits, player, legal_moves):
        res = super()._find_immediate_win(board, hash_key, bits, player, legal_moves)
        if self._rec is not None and res is not None and self._rec.get("stage") is None:
            self._rec["stage"] = "immediate_win" if player == self.player_id else "immediate_block"
        return res

    # ------------------------------------------------------------------
    # Search-tree capture: wrap the real search; derive the move via board diff.
    # ------------------------------------------------------------------
    def _alpha_beta_root(self, board, hash_key, bits, root_player, candidates, depth, deadline):
        root_node = {"m": None, "d": depth, "t": root_player, "a": None, "b": None,
                     "s": None, "w": "full", "leaf": False, "term": None,
                     "ttcut": False, "cut": False, "pruned": 0, "cand": len(candidates),
                     "root": True, "kids": []}
        self._node_stack = [(root_node, list(board))]
        self._nodes_recorded = 1
        t0 = time.perf_counter()
        nodes_before = self._nodes
        try:
            move, score = super()._alpha_beta_root(board, hash_key, bits, root_player, candidates, depth, deadline)
        except champ.SearchTimeout:
            root_node["timeout"] = True
            self._depth_trees[depth] = root_node
            self._node_stack = []
            raise
        self._node_stack = []
        root_node["s"] = _r(score)
        root_node["best"] = (None if move is None else int(move))
        self._mark_pv(root_node, move)
        self._depth_trees[depth] = root_node
        self._last_completed_depth = depth
        self._rec["id"].append({
            "depth": depth,
            "move": (None if move is None else int(move)),
            "score": _r(score),
            "nodes": int(self._nodes - nodes_before),
            "time": round(time.perf_counter() - t0, 3),
        })
        return move, score

    def _alpha_beta(self, board, hash_key, bits, turn, root_player, depth, alpha, beta, deadline):
        node = None
        record = bool(self._node_stack) and self._nodes_recorded < self.NODE_RECORD_BUDGET
        if record:
            parent_node, parent_board = self._node_stack[-1]
            mv = None
            for i in range(self._n2):
                if board[i] != parent_board[i]:
                    mv = i
                    break
            width = beta - alpha
            node = {"m": (None if mv is None else int(mv)), "d": depth, "t": turn,
                    "a": _r(alpha), "b": _r(beta), "s": None,
                    "w": "null" if width <= self.PVS_EPSILON * 2.0 else "full",
                    "leaf": False, "term": None, "ttcut": False, "cut": False,
                    "pruned": 0, "cand": None, "kids": []}
            parent_node["kids"].append(node)
            self._node_stack.append((node, list(board)))
            self._nodes_recorded += 1
        try:
            score = super()._alpha_beta(board, hash_key, bits, turn, root_player, depth, alpha, beta, deadline)
        finally:
            if record:
                self._node_stack.pop()
        if node is not None:
            node["s"] = _r(score)
            if abs(score) >= _WIN:
                node["term"] = "win" if score > 0 else "loss"
            if depth <= 0:
                node["leaf"] = True
            if not node["kids"] and depth > 0 and node["term"] is None and not node["leaf"]:
                node["ttcut"] = True
            if node["cand"] and node["kids"]:
                distinct = len({k["m"] for k in node["kids"]})
                if distinct < node["cand"]:
                    node["cut"] = True
                    node["pruned"] = int(node["cand"] - distinct)
        return score

    def _ordered_candidates(self, board, hash_key, turn, root_player, limit):
        res = super()._ordered_candidates(board, hash_key, turn, root_player, limit)
        if self._node_stack:
            self._node_stack[-1][0]["cand"] = len(res)
        return res

    def _mark_pv(self, root_node, best_move):
        """Highlight the principal variation: follow best child (by score) down."""
        node = root_node
        # root: pick the child whose move == best_move (full-window)
        maximizing = True
        while node is not None and node["kids"]:
            kids = [k for k in node["kids"] if k["w"] == "full"] or node["kids"]
            if node is root_node and best_move is not None:
                pv = next((k for k in kids if k["m"] == best_move), None)
            else:
                if maximizing:
                    pv = max(kids, key=lambda k: (k["s"] if k["s"] is not None else -1e18))
                else:
                    pv = min(kids, key=lambda k: (k["s"] if k["s"] is not None else 1e18))
            if pv is None:
                break
            pv["pv"] = True
            node = pv
            maximizing = not maximizing

    # ------------------------------------------------------------------
    # Per-position analyses (computed once per move, full fidelity)
    # ------------------------------------------------------------------
    def _resistance_detail(self, barr, player):
        n2 = self._n2
        w_own = self.RES_W_OWN
        W = np.where(barr == player, w_own, np.where(barr == EMPTY, 1.0, 0.0))
        I = self._res_I
        J = self._res_J
        g = W[I] * W[J]
        A = np.zeros((n2, n2))
        A[I, J] = -g
        A[J, I] = -g
        deg = np.bincount(I, weights=g, minlength=n2) + np.bincount(J, weights=g, minlength=n2)
        if player == PLAYER_0:
            s = np.where(self._res_start0, W, 0.0)
            t = np.where(self._res_end0, W, 0.0)
        else:
            s = np.where(self._res_start1, W, 0.0)
            t = np.where(self._res_end1, W, 0.0)
        A[self._res_diag] = deg + s + t + self.RES_EPS
        opp = 1 - player
        bs, be, ba, bb = self._br_src, self._br_end, self._br_ca, self._br_cb
        valid = (barr[bs] == player) & (barr[be] == player) & (barr[ba] != opp) & (barr[bb] != opp)
        bridges = []
        if valid.any():
            si = bs[valid]
            ei = be[valid]
            gb = self.RES_BRIDGE_COND
            np.add.at(A, (si, ei), -gb)
            np.add.at(A, (ei, si), -gb)
            np.add.at(A, (si, si), gb)
            np.add.at(A, (ei, ei), gb)
            bridges = [[int(a), int(b)] for a, b in zip(si.tolist(), ei.tolist())]
        try:
            v = np.linalg.solve(A, s)
        except np.linalg.LinAlgError:
            v = np.linalg.lstsq(A, s, rcond=None)[0]
        C = float(s @ (1.0 - v))
        R = (1.0 / C) if C > 1e-9 else self.RES_MAX
        cur = g * (v[I] - v[J])
        Il, Jl, curl, gl = I.tolist(), J.tolist(), cur.tolist(), g.tolist()
        edges = []
        for k in range(len(Il)):
            if gl[k] > 0.0 and abs(curl[k]) > 1e-3:
                edges.append([Il[k], Jl[k], round(curl[k], 4)])
        return {
            "R": (round(R, 5) if R < _INFCAP else None),
            "v": [round(float(x), 4) for x in v.tolist()],
            "edges": edges,
            "bridges": bridges,
        }

    def _dijkstra_detail(self, board, hash_key):
        out = {}
        for pl in (0, 1):
            dist, path = self._distance_and_path(board, hash_key, pl)
            out[str(pl)] = {"dist": (round(float(dist), 3) if dist < _INFCAP else None),
                            "path": [int(c) for c in path]}
        return out

    def _ranking_detail(self, board, hash_key, bits, player):
        opp = self._other(player)
        my_before, my_path = self._distance_and_path(board, hash_key, player)
        opp_before, opp_path = self._distance_and_path(board, hash_key, opp)
        stones = self._stone_count(board)
        n = self._cached_size or 11
        my_set, opp_set = set(my_path), set(opp_path)
        save = self._bridge_save_moves(board, player)
        rows = []
        for move in self._legal_moves(board):
            new_hash = self._hash_after_play(hash_key, move, player)
            board[move] = player
            my_after, _ = self._distance_and_path(board, new_hash, player)
            opp_after, _ = self._distance_and_path(board, new_hash, opp)
            board[move] = EMPTY
            w = 13.5 if player == PLAYER_1 else 10.5
            comp = {
                "race": round(17.0 * self._bounded_delta(my_before, my_after), 2),
                "disrupt": round(w * self._bounded_delta(opp_after, opp_before), 2),
                "static": round(self._static_move_score(board, move, player), 2),
                "save": (self.SAVE_BRIDGE_BONUS if move in save else 0.0),
                "on_my_path": (7.0 if move in my_set else 0.0),
                "on_opp_path": (9.0 if move in opp_set else 0.0),
                "near_my_path": (2.2 if self._is_neighbor_of_set(move, my_set) else 0.0),
                "near_opp_path": (3.4 if self._is_neighbor_of_set(move, opp_set) else 0.0),
            }
            r, c = self._row[move], self._col[move]
            edge = 0.0
            if stones < 18 and (r == 0 or c == 0 or r == n - 1 or c == n - 1):
                edge -= 8.0
            if stones < 24 and (r, c) in {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}:
                edge -= 5.0
            comp["edge"] = edge
            rows.append([move, sum(comp.values()), comp])
        rows.sort(key=lambda x: x[1], reverse=True)
        keep = max(self.ROOT_CANDIDATE_LIMIT, 24)
        big_deadline = time.perf_counter() + 30.0
        out = []
        for move, subtotal, comp in rows[:keep]:
            danger = self._root_danger_score_after_move(board, hash_key, bits, move, player, big_deadline)
            comp["danger"] = round(-danger, 2)
            out.append({"cell": int(move), "total": round(subtotal - danger, 2), "comp": comp})
        out.sort(key=lambda x: x["total"], reverse=True)
        return out

    def _immediate_win_cells(self, board, bits, player):
        own = bits[player]
        cells = []
        for mv in self._legal_moves(board):
            if self._has_bits_won(self._replace_player_bits(bits, player, own | self._bit[mv]), player):
                cells.append(int(mv))
        return cells

    def _formed_bridges(self, board, player):
        opp = 1 - player
        out = []
        for idx in range(self._n2):
            if board[idx] != player:
                continue
            for endpoint, ca, cb in self._bridge_patterns_by_cell[idx]:
                if endpoint <= idx:
                    continue
                if board[endpoint] == player and board[ca] != opp and board[cb] != opp:
                    out.append({"a": int(idx), "b": int(endpoint), "c": [int(ca), int(cb)]})
        return out

    def _tactics_detail(self, board, hash_key, bits, player):
        opp = self._other(player)
        return {
            "immediate_win": self._immediate_win_cells(board, bits, player),
            "immediate_block": self._immediate_win_cells(board, bits, opp),
            "save_bridge": sorted(int(c) for c in self._bridge_save_moves(board, player)),
            "bridges": {"0": self._formed_bridges(board, 0), "1": self._formed_bridges(board, 1)},
        }

    def _finalize_trace(self, state, move):
        n = int(state.board_size)
        self._ensure_board_tools(n)
        board = self._flatten_board(state.board)
        hash_key = self._hash_board(board)
        bits = self._bits_from_board(board)
        player = int(state.turn)
        barr = np.asarray(board, dtype=np.int8)
        rec = self._rec
        if rec.get("stage") is None:
            rec["stage"] = "search"
        rec["n"] = n
        rec["player"] = player
        rec["board"] = [int(x) for x in board]
        rec["chosen"] = int(move.position[0]) * n + int(move.position[1])
        rec["resistance"] = {"0": self._resistance_detail(barr, 0),
                             "1": self._resistance_detail(barr, 1)}
        r0 = rec["resistance"]["0"]["R"] or self.RES_MAX
        r1 = rec["resistance"]["1"]["R"] or self.RES_MAX
        r_me, r_opp = (r0, r1) if player == 0 else (r1, r0)
        rec["eval_score"] = round(self.RES_SCALE * (math.log(r_opp) - math.log(r_me)), 2)
        rec["dijkstra"] = self._dijkstra_detail(board, hash_key)
        rec["ranking"] = self._ranking_detail(board, hash_key, bits, player)
        rec["tactics"] = self._tactics_detail(board, hash_key, bits, player)
        rec["tree"] = self._depth_trees.get(self._last_completed_depth)
        rec["tree_depth"] = int(self._last_completed_depth)
        rec["nodes"] = int(self._nodes)
        rec["tt_size"] = int(len(self._tt))
        rec["time"] = round(time.perf_counter() - self._move_t0, 3)
