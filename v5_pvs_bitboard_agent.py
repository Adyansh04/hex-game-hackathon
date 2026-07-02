"""
ch5_agent.py
Optimized Bitboard Hex agent for the AICA Game AI Platform.

Fixes & Upgrades:
1. Fixed the Bitboard Cylinder Bug: Bit shifts now correctly mask edges BEFORE 
   shifting, completely eliminating "phantom connections" across the board.
2. Removed try-finally overhead: Stripped out safe-unwinding in the recursive 
   tree to bypass Python's context-management overhead, relying on the root 
   sandbox to catch timeouts safely.
3. Enhanced Time Checks: Increased check interval to 256 nodes to minimize 
   system clock polling overhead.
"""

import heapq
import random
import time
from dataclasses import dataclass
from typing import override

from gamelib.hex.agent import Agent
from gamelib.hex.gamestate import GameState as State
from gamelib.hex.move import Move

EMPTY = -1
PLAYER_0 = 0
PLAYER_1 = 1
INF = 10**9

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

class SearchTimeout(Exception):
    pass

@dataclass(slots=True)
class TTEntry:
    depth: int
    score: float
    flag: int
    best_move: int | None

class HexAgent(Agent):
    
    MOVE_TIME_LIMIT_SECONDS = 4.35
    ROOT_CANDIDATE_LIMIT = 32
    NODE_CANDIDATE_LIMIT = 14
    EXTRA_LOCAL_CANDIDATES = 10
    MAX_SEARCH_DEPTH = 10
    TWO_PLY_REPLY_LIMIT = 18
    CHECK_INTERVAL = 256 # Increased to reduce time.perf_counter() overhead
    DEBUG = True

    NEIGHBOR_DIRS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
    CYCLIC_DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

    # Bitboard Masks
    LEFT_MASK = sum(1 << (r * 11) for r in range(11))
    RIGHT_MASK = sum(1 << (r * 11 + 10) for r in range(11))
    TOP_MASK = sum(1 << c for c in range(11))
    BOTTOM_MASK = sum(1 << (10 * 11 + c) for c in range(11))
    NOT_LEFT_MASK = ~LEFT_MASK
    NOT_RIGHT_MASK = ~RIGHT_MASK

    @override
    def initialize(self, init_data: dict) -> None:
        self.player_id = int(init_data["player_id"])
        self.opponent_id = 1 - self.player_id
        self.rng = random.Random(1_431_993 + 97_531 * self.player_id)

        self._cached_size: int | None = None
        self._n2 = 0
        self._neighbors: list[list[int]] = []
        self._row: list[int] = []
        self._col: list[int] = []
        self._bridge_patterns_by_cell: list[list[tuple[int, int, int]]] = []
        self._center_score: list[float] = []

        self._dist: list[float] = []
        self._parent: list[int] = []
        self._zobrist: list[list[int]] = []

        self._distance_cache: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
        self._tt: dict[tuple[int, int], TTEntry] = {}
        self._killer_moves: list[list[int]] = []
        self._nodes = 0
        
        self._p0_bits = 0
        self._p1_bits = 0

    @override
    def get_move(self, state: State) -> Move:
        start = time.perf_counter()
        deadline = start + self.MOVE_TIME_LIMIT_SECONDS

        n = int(state.board_size)
        self._ensure_board_tools(n)

        board = self._flatten_board(state.board)
        hash_key = self._hash_board(board)
        
        self._p0_bits = 0
        self._p1_bits = 0
        for idx, val in enumerate(board):
            if val == PLAYER_0:
                self._p0_bits |= (1 << idx)
            elif val == PLAYER_1:
                self._p1_bits |= (1 << idx)

        self._distance_cache.clear()
        self._tt.clear()
        self._killer_moves = [[-1, -1] for _ in range(self.MAX_SEARCH_DEPTH + 1)]
        self._nodes = 0

        player = self.player_id
        if int(state.turn) != player:
            player = int(state.turn)
        opponent = 1 - player

        legal_moves = self._legal_moves(board)
        if not legal_moves:
            raise ValueError("No valid moves available.")

        opening = self._opening_move(board, player, legal_moves)
        if opening is not None:
            return self._make_move(player, opening)

        winning_move = self._find_immediate_win(board, player, legal_moves)
        if winning_move is not None:
            return self._make_move(player, winning_move)

        blocking_move = self._find_immediate_win(board, opponent, legal_moves)
        if blocking_move is not None:
            return self._make_move(player, blocking_move)

        ranked = self._rank_root_moves(board, hash_key, player, legal_moves, deadline)
        best_fallback = ranked[0][0] if ranked else legal_moves[0]

        if time.perf_counter() >= deadline - 0.08:
            return self._make_move(player, best_fallback)

        root_candidates = [move for move, _score in ranked[: self.ROOT_CANDIDATE_LIMIT]]
        best_move = best_fallback
        best_score = -float(INF)
        prev_score = best_score
        completed_depth = 0
        dynamic_deadline = deadline

        try:
            for depth in range(1, self.MAX_SEARCH_DEPTH + 1):
                if time.perf_counter() >= dynamic_deadline - 0.08:
                    break
                    
                move, score = self._alpha_beta_root(board, hash_key, player, root_candidates, depth, dynamic_deadline)
                
                if move is not None:
                    best_move = move
                    best_score = score
                    completed_depth = depth
                
                if depth >= 5 and best_score < prev_score - 100.0:
                    dynamic_deadline = min(start + 4.85, dynamic_deadline + 0.4)
                    
                prev_score = best_score
                if abs(score) >= 900_000:
                    break
        except SearchTimeout:
            pass

        if self.DEBUG:
            elapsed = time.perf_counter() - start
            print(
                f"Move decision took {elapsed:.2f}s | legal={len(legal_moves)} "
                f"depth={completed_depth} nodes={self._nodes} "
                f"tt={len(self._tt)} score={best_score:.2f} move={self._idx_to_pos(best_move)}"
            )

        return self._make_move(player, best_move)

    def _ensure_board_tools(self, n: int) -> None:
        if self._cached_size == n:
            return
        self._cached_size = n
        self._n2 = n * n
        self._neighbors = [[] for _ in range(self._n2)]
        self._row = [0] * self._n2
        self._col = [0] * self._n2
        self._center_score = [0.0] * self._n2

        center = (n - 1) / 2.0
        for r in range(n):
            for c in range(n):
                idx = r * n + c
                self._row[idx] = r
                self._col[idx] = c
                self._center_score[idx] = max(0.0, 7.5 - (abs(r - center) + abs(c - center))) * 0.28
                neigh = []
                for dr, dc in self.NEIGHBOR_DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        neigh.append(nr * n + nc)
                self._neighbors[idx] = neigh

        self._bridge_patterns_by_cell = [[] for _ in range(self._n2)]
        for r in range(n):
            for c in range(n):
                idx = r * n + c
                triples: list[tuple[int, int, int]] = []
                for i, first in enumerate(self.CYCLIC_DIRS):
                    second = self.CYCLIC_DIRS[(i + 1) % len(self.CYCLIC_DIRS)]
                    er, ec = r + first[0] + second[0], c + first[1] + second[1]
                    ar, ac = r + first[0], c + first[1]
                    br, bc = r + second[0], c + second[1]
                    if 0 <= er < n and 0 <= ec < n and 0 <= ar < n and 0 <= ac < n and 0 <= br < n and 0 <= bc < n:
                        triples.append((er * n + ec, ar * n + ac, br * n + bc))
                self._bridge_patterns_by_cell[idx] = triples

        self._dist = [float(INF)] * self._n2
        self._parent = [-1] * self._n2

        zrng = random.Random(8_675_309 + 31_337 * n)
        self._zobrist = [
            [zrng.getrandbits(64) for _ in range(self._n2)],
            [zrng.getrandbits(64) for _ in range(self._n2)],
        ]

    def _flatten_board(self, board_2d: list[list[int]]) -> list[int]:
        return [cell for row in board_2d for cell in row]

    def _hash_board(self, board: list[int]) -> int:
        h = 0
        for idx, value in enumerate(board):
            if value != EMPTY:
                h ^= self._zobrist[value][idx]
        return h

    def _hash_after_play(self, hash_key: int, idx: int, player: int) -> int:
        return hash_key ^ self._zobrist[player][idx]

    def _make_move(self, player: int, idx: int) -> Move:
        r, c = self._idx_to_pos(idx)
        return Move(player=player, position=[r, c])

    def _idx_to_pos(self, idx: int) -> tuple[int, int]:
        return self._row[idx], self._col[idx]

    def _play(self, board: list[int], idx: int, player: int) -> None:
        board[idx] = player
        if player == PLAYER_0:
            self._p0_bits |= (1 << idx)
        else:
            self._p1_bits |= (1 << idx)

    def _undo(self, board: list[int], idx: int, player: int) -> None:
        board[idx] = EMPTY
        if player == PLAYER_0:
            self._p0_bits &= ~(1 << idx)
        else:
            self._p1_bits &= ~(1 << idx)

    def _legal_moves(self, board: list[int]) -> list[int]:
        return [idx for idx, value in enumerate(board) if value == EMPTY]

    def _stone_count(self, board: list[int]) -> int:
        return sum(1 for value in board if value != EMPTY)

    # ------------------------------------------------------------------
    # FIXED: BITBOARD WIN DETECTION 
    # ------------------------------------------------------------------
    def _has_player_won(self, player: int) -> bool:
        """
        Bitwise flood fill. Edges are perfectly masked BEFORE shifts 
        to prevent cylinder wrapping hallucinations.
        """
        if player == PLAYER_0:
            bits = self._p0_bits
            x = bits & self.LEFT_MASK
            if not x: return False
            while True:
                old = x
                x |= (x & self.NOT_LEFT_MASK) >> 1
                x |= (x & self.NOT_RIGHT_MASK) << 1
                x |= x >> 11
                x |= x << 11
                x |= (x & self.NOT_RIGHT_MASK) >> 10
                x |= (x & self.NOT_LEFT_MASK) << 10
                x &= bits
                if x == old: break
            return (x & self.RIGHT_MASK) != 0
        else:
            bits = self._p1_bits
            x = bits & self.TOP_MASK
            if not x: return False
            while True:
                old = x
                x |= (x & self.NOT_LEFT_MASK) >> 1
                x |= (x & self.NOT_RIGHT_MASK) << 1
                x |= x >> 11
                x |= x << 11
                x |= (x & self.NOT_RIGHT_MASK) >> 10
                x |= (x & self.NOT_LEFT_MASK) << 10
                x &= bits
                if x == old: break
            return (x & self.BOTTOM_MASK) != 0

    def _find_immediate_win(self, board: list[int], player: int, legal_moves: list[int]) -> int | None:
        for move in legal_moves:
            self._play(board, move, player)
            won = self._has_player_won(player)
            self._undo(board, move, player)
            if won:
                return move
        return None

    def _count_immediate_wins(self, board: list[int], player: int, legal_moves: list[int], cap: int = 2) -> int:
        count = 0
        for move in legal_moves:
            if board[move] != EMPTY:
                continue
            self._play(board, move, player)
            won = self._has_player_won(player)
            self._undo(board, move, player)
            if won:
                count += 1
                if count >= cap:
                    return count
        return count

    # ------------------------------------------------------------------
    # OPENING & ROOT LOGIC
    # ------------------------------------------------------------------
    def _opening_move(self, board: list[int], player: int, legal_moves: list[int]) -> int | None:
        n = self._cached_size or 11
        if n % 2 == 0: return None
        center = (n // 2) * n + (n // 2)
        stones = self._stone_count(board)
        legal_set = set(legal_moves)

        if stones == 0 and player == PLAYER_0 and center in legal_set:
            return center

        if stones == 1 and player == PLAYER_1:
            cr, cc = n // 2, n // 2
            ring = [(cr, cc + 1), (cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr - 1, cc + 1), (cr + 1, cc - 1)]
            choices = [r * n + c for r, c in ring if 0 <= r < n and 0 <= c < n and r * n + c in legal_set]
            if choices:
                choices.sort(key=lambda mv: self._static_move_score(board, mv, player), reverse=True)
                return choices[0]
            if center in legal_set:
                return center
        return None

    def _distance_and_path(self, board: list[int], hash_key: int, player: int) -> tuple[float, tuple[int, ...]]:
        cached = self._distance_cache.get((hash_key, player))
        if cached is not None:
            return cached

        n = self._cached_size or 11
        opponent = 1 - player
        dist = self._dist
        parent = self._parent

        for i in range(self._n2):
            dist[i] = float(INF)
            parent[i] = -1

        heap: list[tuple[float, int]] = []

        def cell_cost(idx: int) -> float:
            value = board[idx]
            if value == player: return 0.0
            if value == EMPTY: return 1.0
            return float(INF)

        starts = [r * n for r in range(n)] if player == PLAYER_0 else [c for c in range(n)]

        for idx in starts:
            cost = cell_cost(idx)
            if cost < INF:
                dist[idx] = cost
                heapq.heappush(heap, (cost, idx))

        best_end = -1
        best_cost = float(INF)
        while heap:
            cur_dist, cur = heapq.heappop(heap)
            if cur_dist != dist[cur]:
                continue
            if (player == PLAYER_0 and self._col[cur] == n - 1) or (player == PLAYER_1 and self._row[cur] == n - 1):
                best_cost = cur_dist
                best_end = cur
                break

            for nxt in self._neighbors[cur]:
                step = cell_cost(nxt)
                if step >= INF: continue
                nd = cur_dist + step
                if nd < dist[nxt]:
                    dist[nxt] = nd
                    parent[nxt] = cur
                    heapq.heappush(heap, (nd, nxt))

            if board[cur] == player:
                for endpoint, carrier_a, carrier_b in self._bridge_patterns_by_cell[cur]:
                    if board[endpoint] == player and board[carrier_a] != opponent and board[carrier_b] != opponent:
                        nd = cur_dist + 0.45
                        if nd < dist[endpoint]:
                            dist[endpoint] = nd
                            parent[endpoint] = cur
                            heapq.heappush(heap, (nd, endpoint))

        if best_end < 0:
            result = (float(INF), tuple())
            self._distance_cache[(hash_key, player)] = result
            return result

        path: list[int] = []
        cur = best_end
        while cur >= 0:
            if board[cur] == EMPTY: path.append(cur)
            cur = parent[cur]
        
        result = (best_cost, tuple(path))
        self._distance_cache[(hash_key, player)] = result
        return result

    def _rank_root_moves(self, board: list[int], hash_key: int, player: int, legal_moves: list[int], deadline: float) -> list[tuple[int, float]]:
        opponent = 1 - player
        my_before, my_path = self._distance_and_path(board, hash_key, player)
        opp_before, opp_path = self._distance_and_path(board, hash_key, opponent)
        stones = self._stone_count(board)
        my_path_set = set(my_path)
        opp_path_set = set(opp_path)

        ranked: list[tuple[int, float]] = []
        for move in legal_moves:
            if time.perf_counter() >= deadline - 0.12: break

            new_hash = self._hash_after_play(hash_key, move, player)
            self._play(board, move, player)
            
            my_after, _ = self._distance_and_path(board, new_hash, player)
            opp_after, _ = self._distance_and_path(board, new_hash, opponent)
            
            self._undo(board, move, player)

            my_gain = min(my_before, 50.0) - min(my_after, 50.0)
            opp_disruption = min(opp_before, 50.0) - min(opp_after, 50.0)
            
            score = 17.0 * my_gain + (13.5 if player == PLAYER_1 else 10.5) * opp_disruption
            score += self._static_move_score(board, move, player)

            if move in my_path_set: score += 7.0
            if move in opp_path_set: score += 9.0
            
            r, c = self._row[move], self._col[move]
            n = self._cached_size or 11
            if stones < 18 and (r == 0 or c == 0 or r == n - 1 or c == n - 1): score -= 8.0
            if stones < 24 and (r, c) in {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}: score -= 5.0

            score -= self._root_danger_score(board, hash_key, move, player, deadline)
            ranked.append((move, score))

        if not ranked:
            ranked = [(m, self._static_move_score(board, m, player)) for m in legal_moves]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _root_danger_score(self, board: list[int], hash_key: int, move: int, player: int, deadline: float) -> float:
        opponent = 1 - player
        new_hash = self._hash_after_play(hash_key, move, player)
        self._play(board, move, player)
        
        try:
            remaining = self._legal_moves(board)
            immediate = self._count_immediate_wins(board, opponent, remaining, cap=2)
            if immediate >= 2: return 600_000.0
            if immediate == 1: return 250_000.0

            opp_dist, opp_path = self._distance_and_path(board, new_hash, opponent)
            if opp_dist > 3 and len(opp_path) > 4: return 0.0

            replies = self._ordered_candidates(board, new_hash, opponent, opponent, self.TWO_PLY_REPLY_LIMIT, depth=0)
            for cell in opp_path:
                if board[cell] == EMPTY and cell not in replies:
                    replies.append(cell)

            worst = 0.0
            for reply in replies[:max(self.TWO_PLY_REPLY_LIMIT, len(opp_path))]:
                if time.perf_counter() >= deadline - 0.10: break
                if board[reply] != EMPTY: continue
                
                self._play(board, reply, opponent)
                try:
                    if self._has_player_won(opponent): return 500_000.0
                    after_legal = self._legal_moves(board)
                    fork_count = self._count_immediate_wins(board, opponent, after_legal, cap=2)
                    if fork_count >= 2: return 300_000.0
                    if fork_count == 1: worst = max(worst, 5_000.0)
                finally:
                    self._undo(board, reply, opponent)
            return worst
        finally:
            self._undo(board, move, player)

    # ------------------------------------------------------------------
    # PVS + ALPHA BETA SEARCH (Try/Finally Removed for Speed)
    # ------------------------------------------------------------------
    def _alpha_beta_root(self, board: list[int], hash_key: int, root_player: int, candidates: list[int], depth: int, deadline: float) -> tuple[int | None, float]:
        alpha = -float(INF)
        beta = float(INF)
        best_move: int | None = None
        best_score = -float(INF)

        root_key = (hash_key, root_player)
        tt_move = self._tt.get(root_key).best_move if self._tt.get(root_key) else None
        ordered = candidates[:]
        if tt_move in ordered:
            ordered.remove(tt_move)
            ordered.insert(0, tt_move)

        for i, move in enumerate(ordered):
            self._check_time(deadline, force=True)
            if board[move] != EMPTY: continue
            
            child_hash = self._hash_after_play(hash_key, move, root_player)
            self._play(board, move, root_player)
            
            if self._has_player_won(root_player):
                score = 1_000_000.0 + depth
            else:
                if i == 0:
                    score = self._alpha_beta(board, child_hash, 1 - root_player, root_player, depth - 1, alpha, beta, deadline)
                else:
                    score = self._alpha_beta(board, child_hash, 1 - root_player, root_player, depth - 1, alpha, alpha + 1, deadline)
                    if alpha < score < beta:
                        score = self._alpha_beta(board, child_hash, 1 - root_player, root_player, depth - 1, score, beta, deadline)
            
            self._undo(board, move, root_player)

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

        return best_move, best_score

    def _alpha_beta(self, board: list[int], hash_key: int, turn: int, root_player: int, depth: int, alpha: float, beta: float, deadline: float) -> float:
        self._nodes += 1
        self._check_time(deadline)

        alpha_orig = alpha
        beta_orig = beta
        key = (hash_key, turn)
        entry = self._tt.get(key)
        
        if entry is not None and entry.depth >= depth:
            if entry.flag == TT_EXACT: return entry.score
            if entry.flag == TT_LOWER: alpha = max(alpha, entry.score)
            elif entry.flag == TT_UPPER: beta = min(beta, entry.score)
            if alpha >= beta: return entry.score

        opponent = 1 - root_player
        if self._has_player_won(root_player): return 1_000_000.0 + depth
        if self._has_player_won(opponent): return -1_000_000.0 - depth
        if depth <= 0: return self._evaluate_board(board, hash_key, root_player)

        candidates = self._ordered_candidates(board, hash_key, turn, root_player, self.NODE_CANDIDATE_LIMIT, depth)
        if not candidates: return self._evaluate_board(board, hash_key, root_player)

        ordered = []
        if entry and entry.best_move in candidates:
            ordered.append(entry.best_move)
            candidates.remove(entry.best_move)
            
        k1, k2 = self._killer_moves[depth]
        if k1 in candidates:
            ordered.append(k1)
            candidates.remove(k1)
        if k2 in candidates:
            ordered.append(k2)
            candidates.remove(k2)
        ordered.extend(candidates)

        best_move: int | None = None
        
        if turn == root_player: # MAXIMIZER
            value = -float(INF)
            for i, move in enumerate(ordered):
                child_hash = self._hash_after_play(hash_key, move, turn)
                self._play(board, move, turn)
                
                if i == 0:
                    child_score = self._alpha_beta(board, child_hash, 1 - turn, root_player, depth - 1, alpha, beta, deadline)
                else:
                    child_score = self._alpha_beta(board, child_hash, 1 - turn, root_player, depth - 1, alpha, alpha + 1, deadline)
                    if alpha < child_score < beta:
                        child_score = self._alpha_beta(board, child_hash, 1 - turn, root_player, depth - 1, child_score, beta, deadline)
                
                self._undo(board, move, turn)
                    
                if child_score > value:
                    value = child_score
                    best_move = move
                if value > alpha: alpha = value
                if alpha >= beta:
                    if best_move != self._killer_moves[depth][0]:
                        self._killer_moves[depth][1] = self._killer_moves[depth][0]
                        self._killer_moves[depth][0] = best_move
                    break
        else: # MINIMIZER
            value = float(INF)
            for i, move in enumerate(ordered):
                child_hash = self._hash_after_play(hash_key, move, turn)
                self._play(board, move, turn)
                
                if i == 0:
                    child_score = self._alpha_beta(board, child_hash, 1 - turn, root_player, depth - 1, alpha, beta, deadline)
                else:
                    child_score = self._alpha_beta(board, child_hash, 1 - turn, root_player, depth - 1, beta - 1, beta, deadline)
                    if alpha < child_score < beta:
                        child_score = self._alpha_beta(board, child_hash, 1 - turn, root_player, depth - 1, alpha, child_score, deadline)
                
                self._undo(board, move, turn)
                    
                if child_score < value:
                    value = child_score
                    best_move = move
                if value < beta: beta = value
                if alpha >= beta:
                    if best_move != self._killer_moves[depth][0]:
                        self._killer_moves[depth][1] = self._killer_moves[depth][0]
                        self._killer_moves[depth][0] = best_move
                    break

        if value <= alpha_orig: flag = TT_UPPER
        elif value >= beta_orig: flag = TT_LOWER
        else: flag = TT_EXACT
        
        self._tt[key] = TTEntry(depth=depth, score=value, flag=flag, best_move=best_move)
        return value

    def _ordered_candidates(self, board: list[int], hash_key: int, turn: int, root_player: int, limit: int, depth: int) -> list[int]:
        legal = self._legal_moves(board)
        if not legal: return []

        opponent = 1 - turn
        _, turn_path = self._distance_and_path(board, hash_key, turn)
        _, opp_path = self._distance_and_path(board, hash_key, opponent)

        candidate_set: set[int] = set()
        for path in (turn_path, opp_path):
            for cell in path:
                if board[cell] == EMPTY: candidate_set.add(cell)
                for nbr in self._neighbors[cell]:
                    if board[nbr] == EMPTY: candidate_set.add(nbr)

        local_sorted = sorted(legal, key=lambda mv: self._static_move_score(board, mv, turn), reverse=True)
        for mv in local_sorted[: self.EXTRA_LOCAL_CANDIDATES]:
            candidate_set.add(mv)

        if not candidate_set: candidate_set.update(local_sorted[:limit])

        turn_set = set(turn_path)
        opp_set = set(opp_path)
        scored: list[tuple[int, float]] = []
        for mv in candidate_set:
            if board[mv] != EMPTY: continue
            score = self._static_move_score(board, mv, turn)
            if mv in turn_set: score += 7.0
            if mv in opp_set: score += 9.0
            scored.append((mv, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [mv for mv, _score in scored[:limit]]

    def _evaluate_board(self, board: list[int], hash_key: int, root_player: int) -> float:
        opponent = 1 - root_player
        root_dist, root_path = self._distance_and_path(board, hash_key, root_player)
        opp_dist, opp_path = self._distance_and_path(board, hash_key, opponent)

        if root_dist == 0: return 1_000_000.0
        if opp_dist == 0: return -1_000_000.0

        score = 120.0 * (min(opp_dist, 50.0) - min(root_dist, 50.0))
        score += 2.0 * (len(opp_path) - len(root_path))

        root_local, opp_local = 0.0, 0.0
        for idx, value in enumerate(board):
            if value == root_player:
                root_local += sum(1 for nbr in self._neighbors[idx] if board[nbr] == root_player)
            elif value == opponent:
                opp_local += sum(1 for nbr in self._neighbors[idx] if board[nbr] == opponent)
        
        score += 0.35 * (root_local - opp_local)
        return score

    def _static_move_score(self, board: list[int], move: int, player: int) -> float:
        opponent = 1 - player
        own_n, opp_n, emp_n = 0, 0, 0
        for nbr in self._neighbors[move]:
            val = board[nbr]
            if val == player: own_n += 1
            elif val == opponent: opp_n += 1
            else: emp_n += 1
            
        score = 1.05 * own_n + 0.62 * opp_n + 0.09 * emp_n + self._center_score[move]
        
        for endpoint, ca, cb in self._bridge_patterns_by_cell[move]:
            if board[endpoint] == player and board[ca] != opponent and board[cb] != opponent:
                score += 2.6
                
        n = self._cached_size or 11
        nearest = min(self._col[move], n - 1 - self._col[move]) if player == PLAYER_0 else min(self._row[move], n - 1 - self._row[move])
        score += 0.08 * (n / 2.0 - nearest)
        
        return score

    def _check_time(self, deadline: float, force: bool = False) -> None:
        if force or (self._nodes & (self.CHECK_INTERVAL - 1)) == 0:
            if time.perf_counter() >= deadline - 0.06:
                raise SearchTimeout()

if __name__ == "__main__":
    agent = HexAgent()
    agent.start()