"""
Final competitive Hex agent for the AICA Game AI Platform.

Key design choices
------------------
1. Uses the verified platform mapping:
      Player 0 connects LEFT  -> RIGHT.
      Player 1 connects TOP   -> BOTTOM.
2. Never prunes one-move tactics: immediate win and immediate block scan all empty cells.
3. Adds a root two-ply threat guard to avoid moves that allow opponent double threats.
4. Uses path-focused iterative deepening alpha-beta as the main search, because it is
   more reliable than generic MCTS under the 5-second Python limit.
5. Keeps all search mutations on a local board copy and always undoes moves safely.

Only Python standard library + gamelib are used.
"""

from __future__ import annotations

from collections import deque
import heapq
import math
import random
import time
from typing import override

from gamelib.hex.agent import Agent
from gamelib.hex.gamestate import GameState as State
from gamelib.hex.move import Move


EMPTY = -1
PLAYER_0 = 0
PLAYER_1 = 1
INF = 10**9


class SearchTimeout(Exception):
    """Raised internally when the search reaches the wall-clock deadline."""


class HexAgent(Agent):
    """
    Tactical + path-search Hex agent.

    This version is deliberately more adversarial than the earlier MCTS version:
    opponent nodes are handled by minimization, and root moves are filtered/ranked
    by whether they allow immediate or two-ply opponent threats.
    """

    # The platform limit is 5 seconds. Keep a safety margin.
    MOVE_TIME_LIMIT_SECONDS = 4.35

    # Search tuning. These are intentionally conservative for pure Python.
    ROOT_CANDIDATE_LIMIT = 28
    NODE_CANDIDATE_LIMIT = 12
    EXTRA_LOCAL_CANDIDATES = 10
    MAX_SEARCH_DEPTH = 7
    TWO_PLY_REPLY_LIMIT = 18
    DEBUG = True

    # Hex neighbor geometry used by the provided gamelib engine.
    NEIGHBOR_DIRS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
    CYCLIC_DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

    @override
    def initialize(self, init_data: dict) -> None:
        self.player_id = int(init_data["player_id"])
        self.opponent_id = 1 - self.player_id

        # Verified from gamelib.hex.engine:
        #   Player 0 = horizontal / left-right
        #   Player 1 = vertical / top-bottom
        assert self._goal_axis(PLAYER_0) == "horizontal"
        assert self._goal_axis(PLAYER_1) == "vertical"

        self.rng = random.Random(992_831 + self.player_id * 104_729)
        self._cached_size: int | None = None
        self._neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._bridge_patterns: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []

    @override
    def get_move(self, state: State) -> Move:
        """Return a legal move for the current board state."""
        start_time = time.perf_counter()
        deadline = start_time + self.MOVE_TIME_LIMIT_SECONDS

        board = [row[:] for row in state.board]
        n = int(state.board_size)
        self._ensure_board_tools(n)

        player = self.player_id
        # DevRunner/platform should only call the matching agent on its turn. If this
        # ever happens, returning with state.turn is the only way to remain legal.
        if int(state.turn) != player:
            player = int(state.turn)
        opponent = self._other(player)

        legal_moves = self._legal_moves(board)
        if not legal_moves:
            raise ValueError("No valid moves available.")

        opening = self._opening_move(board, player, legal_moves)
        if opening is not None:
            return Move(player=player, position=[opening[0], opening[1]])

        # Full-board tactical layer. No candidate pruning here.
        winning_move = self._find_immediate_win(board, player, legal_moves)
        if winning_move is not None:
            return Move(player=player, position=[winning_move[0], winning_move[1]])

        blocking_move = self._find_immediate_win(board, opponent, legal_moves)
        if blocking_move is not None:
            return Move(player=player, position=[blocking_move[0], blocking_move[1]])

        # Root ranking includes path features and a two-ply fork safety penalty.
        ranked_moves = self._rank_root_moves(board, player, legal_moves, deadline)
        best_fallback = ranked_moves[0][0] if ranked_moves else legal_moves[0]

        if time.perf_counter() >= deadline - 0.08:
            return Move(player=player, position=[best_fallback[0], best_fallback[1]])

        # Path-focused iterative-deepening alpha-beta. This replaces the previous
        # generic MCTS as the primary decision-maker.
        best_move = best_fallback
        best_score = -INF
        completed_depth = 0

        try:
            root_candidates = [move for move, _score in ranked_moves[: self.ROOT_CANDIDATE_LIMIT]]
            for depth in range(1, self.MAX_SEARCH_DEPTH + 1):
                if time.perf_counter() >= deadline - 0.08:
                    break
                move, score = self._alpha_beta_root(board, player, root_candidates, depth, deadline)
                if move is not None:
                    best_move = move
                    best_score = score
                    completed_depth = depth
                if abs(score) >= 900_000:
                    break
        except SearchTimeout:
            pass

        if self.DEBUG:
            elapsed = time.perf_counter() - start_time
            print(
                f"Move decision took {elapsed:.2f}s | legal={len(legal_moves)} "
                f"ranked={len(ranked_moves)} depth={completed_depth} score={best_score:.2f} move={best_move}"
            )

        return Move(player=player, position=[best_move[0], best_move[1]])

    # ------------------------------------------------------------------
    # Board helpers
    # ------------------------------------------------------------------

    def _ensure_board_tools(self, n: int) -> None:
        if self._cached_size == n:
            return

        self._cached_size = n
        self._neighbors = {}
        for r in range(n):
            for c in range(n):
                self._neighbors[(r, c)] = [
                    (r + dr, c + dc)
                    for dr, dc in self.NEIGHBOR_DIRS
                    if 0 <= r + dr < n and 0 <= c + dc < n
                ]

        patterns = []
        for i, first in enumerate(self.CYCLIC_DIRS):
            second = self.CYCLIC_DIRS[(i + 1) % len(self.CYCLIC_DIRS)]
            endpoint = (first[0] + second[0], first[1] + second[1])
            patterns.append((endpoint, first, second))
        self._bridge_patterns = patterns

    @staticmethod
    def _goal_axis(player: int) -> str:
        return "horizontal" if player == PLAYER_0 else "vertical"

    @staticmethod
    def _other(player: int) -> int:
        return 1 - player

    @staticmethod
    def _play(board: list[list[int]], move: tuple[int, int], player: int) -> None:
        board[move[0]][move[1]] = player

    @staticmethod
    def _undo(board: list[list[int]], move: tuple[int, int]) -> None:
        board[move[0]][move[1]] = EMPTY

    def _legal_moves(self, board: list[list[int]]) -> list[tuple[int, int]]:
        return [(r, c) for r, row in enumerate(board) for c, value in enumerate(row) if value == EMPTY]

    def _stone_count(self, board: list[list[int]]) -> int:
        return sum(1 for row in board for value in row if value != EMPTY)

    # ------------------------------------------------------------------
    # Opening
    # ------------------------------------------------------------------

    def _opening_move(
        self,
        board: list[list[int]],
        player: int,
        legal_moves: list[tuple[int, int]],
    ) -> tuple[int, int] | None:
        n = len(board)
        if n % 2 == 0:
            return None
        center = (n // 2, n // 2)
        stones = self._stone_count(board)
        legal_set = set(legal_moves)

        # No swap rule: Player 0 should take exact center immediately.
        if stones == 0 and player == PLAYER_0 and center in legal_set:
            return center

        # Strong response if Player 1 sees center occupied.
        if stones == 1 and player == PLAYER_1:
            ring = [
                (center[0], center[1] + 1),
                (center[0] - 1, center[1]),
                (center[0] + 1, center[1]),
                (center[0], center[1] - 1),
                (center[0] - 1, center[1] + 1),
                (center[0] + 1, center[1] - 1),
            ]
            choices = [mv for mv in ring if mv in legal_set]
            if choices:
                choices.sort(key=lambda mv: self._static_move_score(board, mv, player), reverse=True)
                return choices[0]
            if center in legal_set:
                return center

        return None

    # ------------------------------------------------------------------
    # Win detection and immediate tactics
    # ------------------------------------------------------------------

    def _has_player_won(self, board: list[list[int]], player: int) -> bool:
        n = len(board)
        visited = [[False] * n for _ in range(n)]
        queue: deque[tuple[int, int]] = deque()

        if player == PLAYER_0:
            for r in range(n):
                if board[r][0] == player:
                    visited[r][0] = True
                    queue.append((r, 0))
            while queue:
                r, c = queue.popleft()
                if c == n - 1:
                    return True
                for nr, nc in self._neighbors[(r, c)]:
                    if not visited[nr][nc] and board[nr][nc] == player:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
        else:
            for c in range(n):
                if board[0][c] == player:
                    visited[0][c] = True
                    queue.append((0, c))
            while queue:
                r, c = queue.popleft()
                if r == n - 1:
                    return True
                for nr, nc in self._neighbors[(r, c)]:
                    if not visited[nr][nc] and board[nr][nc] == player:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
        return False

    def _find_immediate_win(
        self,
        board: list[list[int]],
        player: int,
        legal_moves: list[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """Check every legal cell for a one-move win. Never prune this scan."""
        ordered = sorted(
            legal_moves,
            key=lambda mv: self._fast_local_score(board, mv, player),
            reverse=True,
        )
        for move in ordered:
            self._play(board, move, player)
            won = self._has_player_won(board, player)
            self._undo(board, move)
            if won:
                return move
        return None

    def _count_immediate_wins(
        self,
        board: list[list[int]],
        player: int,
        legal_moves: list[tuple[int, int]] | None = None,
        cap: int = 2,
    ) -> int:
        """Count immediate winning moves for player, stopping at cap."""
        if legal_moves is None:
            legal_moves = self._legal_moves(board)
        count = 0
        for move in legal_moves:
            if board[move[0]][move[1]] != EMPTY:
                continue
            self._play(board, move, player)
            won = self._has_player_won(board, player)
            self._undo(board, move)
            if won:
                count += 1
                if count >= cap:
                    return count
        return count

    # ------------------------------------------------------------------
    # Path distance and move scoring
    # ------------------------------------------------------------------

    def _distance_and_path(self, board: list[list[int]], player: int) -> tuple[float, list[tuple[int, int]]]:
        """
        Dijkstra race estimate and one shortest path.

        Own stones cost 0, empty cells cost 1, opponent stones are impassable.
        Returned path contains only empty cells on the best path.
        """
        n = len(board)
        opponent = self._other(player)
        dist = [[INF] * n for _ in range(n)]
        parent: list[list[tuple[int, int] | None]] = [[None] * n for _ in range(n)]
        heap: list[tuple[int, int, int]] = []

        def cost_of(r: int, c: int) -> int:
            value = board[r][c]
            if value == player:
                return 0
            if value == EMPTY:
                return 1
            if value == opponent:
                return INF
            return INF

        if player == PLAYER_0:
            starts = [(r, 0) for r in range(n)]
        else:
            starts = [(0, c) for c in range(n)]

        for r, c in starts:
            cost = cost_of(r, c)
            if cost < INF:
                dist[r][c] = cost
                heapq.heappush(heap, (cost, r, c))

        best_end: tuple[int, int] | None = None
        best_cost = INF
        while heap:
            cur, r, c = heapq.heappop(heap)
            if cur != dist[r][c]:
                continue
            if (player == PLAYER_0 and c == n - 1) or (player == PLAYER_1 and r == n - 1):
                best_cost = cur
                best_end = (r, c)
                break
            for nr, nc in self._neighbors[(r, c)]:
                step = cost_of(nr, nc)
                if step >= INF:
                    continue
                new_dist = cur + step
                if new_dist < dist[nr][nc]:
                    dist[nr][nc] = new_dist
                    parent[nr][nc] = (r, c)
                    heapq.heappush(heap, (new_dist, nr, nc))

        if best_end is None:
            return float(INF), []

        path: list[tuple[int, int]] = []
        cur_cell: tuple[int, int] | None = best_end
        while cur_cell is not None:
            r, c = cur_cell
            if board[r][c] == EMPTY:
                path.append((r, c))
            cur_cell = parent[r][c]
        return float(best_cost), path

    def _rank_root_moves(
        self,
        board: list[list[int]],
        player: int,
        legal_moves: list[tuple[int, int]],
        deadline: float,
    ) -> list[tuple[tuple[int, int], float]]:
        opponent = self._other(player)
        my_before, my_path_before = self._distance_and_path(board, player)
        opp_before, opp_path_before = self._distance_and_path(board, opponent)
        stones = self._stone_count(board)

        ranked: list[tuple[tuple[int, int], float]] = []
        for move in legal_moves:
            if time.perf_counter() >= deadline - 0.12:
                break
            self._play(board, move, player)
            my_after, _ = self._distance_and_path(board, player)
            opp_after, _ = self._distance_and_path(board, opponent)
            self._undo(board, move)

            my_gain = self._bounded_delta(my_before, my_after)
            opp_disruption = self._bounded_delta(opp_after, opp_before)

            # Player 1 receives a stronger defensive bias because no-swap Hex favors P0.
            disruption_weight = 13.5 if player == PLAYER_1 else 10.5
            score = 17.0 * my_gain + disruption_weight * opp_disruption
            score += self._static_move_score(board, move, player)

            # Prefer moves directly related to current race paths.
            if move in my_path_before:
                score += 7.0
            if move in opp_path_before:
                score += 9.0
            if self._is_neighbor_of_any(move, my_path_before):
                score += 2.2
            if self._is_neighbor_of_any(move, opp_path_before):
                score += 3.4

            # Penalize obvious early edge/corner drift. This specifically prevents
            # moves like [0,10] from looking attractive too early.
            r, c = move
            if stones < 18 and (r == 0 or c == 0 or r == len(board) - 1 or c == len(board) - 1):
                score -= 8.0
            if stones < 24 and (r, c) in {(0, 0), (0, len(board) - 1), (len(board) - 1, 0), (len(board) - 1, len(board) - 1)}:
                score -= 5.0

            # Root-only two-ply safety. Huge penalty for moves that allow the opponent
            # to win immediately or create a double immediate threat.
            danger = self._root_danger_score_after_move(board, move, player, deadline)
            score -= danger
            ranked.append((move, score))

        if not ranked:
            ranked = [(move, self._static_move_score(board, move, player)) for move in legal_moves]

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    @staticmethod
    def _bounded_delta(before: float, after: float) -> float:
        return min(before, 50.0) - min(after, 50.0)

    def _is_neighbor_of_any(self, move: tuple[int, int], cells: list[tuple[int, int]]) -> bool:
        if not cells:
            return False
        targets = set(cells)
        return any(neigh in targets for neigh in self._neighbors[move])

    def _static_move_score(self, board: list[list[int]], move: tuple[int, int], player: int) -> float:
        score = self._fast_local_score(board, move, player)
        score += 2.6 * self._bridge_creation_count(board, move, player)
        score += 2.3 * self._opponent_bridge_attack_count(board, move, self._other(player))
        score += self._edge_orientation_bonus(board, move, player)
        return score

    def _fast_local_score(self, board: list[list[int]], move: tuple[int, int], player: int) -> float:
        n = len(board)
        r, c = move
        opponent = self._other(player)
        own_neighbors = 0
        opp_neighbors = 0
        empty_neighbors = 0
        for nr, nc in self._neighbors[(r, c)]:
            value = board[nr][nc]
            if value == player:
                own_neighbors += 1
            elif value == opponent:
                opp_neighbors += 1
            else:
                empty_neighbors += 1

        center = (n - 1) / 2.0
        center_dist = abs(r - center) + abs(c - center)
        center_bonus = max(0.0, 7.5 - center_dist) * 0.28
        return 1.05 * own_neighbors + 0.62 * opp_neighbors + 0.09 * empty_neighbors + center_bonus

    def _edge_orientation_bonus(self, board: list[list[int]], move: tuple[int, int], player: int) -> float:
        n = len(board)
        r, c = move
        nearest_goal_edge = min(c, n - 1 - c) if player == PLAYER_0 else min(r, n - 1 - r)
        return 0.08 * (n / 2.0 - nearest_goal_edge)

    def _bridge_creation_count(self, board: list[list[int]], move: tuple[int, int], player: int) -> int:
        n = len(board)
        r, c = move
        opponent = self._other(player)
        count = 0
        for endpoint_off, carrier_a_off, carrier_b_off in self._bridge_patterns:
            er, ec = r + endpoint_off[0], c + endpoint_off[1]
            ar, ac = r + carrier_a_off[0], c + carrier_a_off[1]
            br, bc = r + carrier_b_off[0], c + carrier_b_off[1]
            if not (0 <= er < n and 0 <= ec < n and 0 <= ar < n and 0 <= ac < n and 0 <= br < n and 0 <= bc < n):
                continue
            if board[er][ec] == player and board[ar][ac] != opponent and board[br][bc] != opponent:
                count += 1
        return count

    def _opponent_bridge_attack_count(self, board: list[list[int]], move: tuple[int, int], opponent: int) -> int:
        n = len(board)
        count = 0
        seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for ar, ac in self._neighbors[move]:
            if board[ar][ac] != opponent:
                continue
            for endpoint_off, carrier_a_off, carrier_b_off in self._bridge_patterns:
                er, ec = ar + endpoint_off[0], ac + endpoint_off[1]
                cr1 = (ar + carrier_a_off[0], ac + carrier_a_off[1])
                cr2 = (ar + carrier_b_off[0], ac + carrier_b_off[1])
                if not (0 <= er < n and 0 <= ec < n):
                    continue
                if board[er][ec] != opponent:
                    continue
                if move != cr1 and move != cr2:
                    continue
                other_carrier = cr2 if move == cr1 else cr1
                orow, ocol = other_carrier
                if 0 <= orow < n and 0 <= ocol < n and board[orow][ocol] == EMPTY:
                    endpoints = tuple(sorted(((ar, ac), (er, ec))))
                    if endpoints not in seen:
                        seen.add(endpoints)
                        count += 1
        return count

    # ------------------------------------------------------------------
    # Two-ply root threat guard
    # ------------------------------------------------------------------

    def _root_danger_score_after_move(
        self,
        board: list[list[int]],
        move: tuple[int, int],
        player: int,
        deadline: float,
    ) -> float:
        """
        Penalize root moves that allow opponent immediate wins or double threats.

        This is designed to catch positions like:
          our move -> opponent reply -> opponent has two separate winning cells.
        """
        opponent = self._other(player)
        self._play(board, move, player)
        try:
            remaining = self._legal_moves(board)
            immediate_count = self._count_immediate_wins(board, opponent, remaining, cap=2)
            if immediate_count >= 2:
                return 600_000.0
            if immediate_count == 1:
                return 250_000.0

            opp_dist, opp_path = self._distance_and_path(board, opponent)
            # Only perform the expensive fork check when opponent is close, or when
            # their path is short enough that a fork is plausible.
            if opp_dist > 3 and len(opp_path) > 4:
                return 0.0

            replies = self._ordered_candidates(board, opponent, opponent, self.TWO_PLY_REPLY_LIMIT)
            # Ensure the shortest path cells are included even if ranking misses them.
            for cell in opp_path:
                if cell not in replies and board[cell[0]][cell[1]] == EMPTY:
                    replies.append(cell)

            worst = 0.0
            for reply in replies[: max(self.TWO_PLY_REPLY_LIMIT, len(opp_path))]:
                if time.perf_counter() >= deadline - 0.10:
                    break
                if board[reply[0]][reply[1]] != EMPTY:
                    continue
                self._play(board, reply, opponent)
                try:
                    if self._has_player_won(board, opponent):
                        return 500_000.0
                    after_reply_legal = self._legal_moves(board)
                    fork_count = self._count_immediate_wins(board, opponent, after_reply_legal, cap=2)
                    if fork_count >= 2:
                        return 300_000.0
                    if fork_count == 1:
                        worst = max(worst, 5_000.0)
                finally:
                    self._undo(board, reply)
            return worst
        finally:
            self._undo(board, move)

    # ------------------------------------------------------------------
    # Alpha-beta search
    # ------------------------------------------------------------------

    def _alpha_beta_root(
        self,
        board: list[list[int]],
        root_player: int,
        candidates: list[tuple[int, int]],
        depth: int,
        deadline: float,
    ) -> tuple[tuple[int, int] | None, float]:
        alpha = -INF
        beta = INF
        best_move: tuple[int, int] | None = None
        best_score = -INF

        for move in candidates:
            self._check_time(deadline)
            if board[move[0]][move[1]] != EMPTY:
                continue
            self._play(board, move, root_player)
            try:
                if self._has_player_won(board, root_player):
                    score = 1_000_000 + depth
                else:
                    score = self._alpha_beta(
                        board,
                        turn=self._other(root_player),
                        root_player=root_player,
                        depth=depth - 1,
                        alpha=alpha,
                        beta=beta,
                        deadline=deadline,
                    )
            finally:
                self._undo(board, move)

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)

        return best_move, best_score

    def _alpha_beta(
        self,
        board: list[list[int]],
        turn: int,
        root_player: int,
        depth: int,
        alpha: float,
        beta: float,
        deadline: float,
    ) -> float:
        self._check_time(deadline)
        opponent = self._other(root_player)

        if self._has_player_won(board, root_player):
            return 1_000_000 + depth
        if self._has_player_won(board, opponent):
            return -1_000_000 - depth
        if depth <= 0:
            return self._evaluate_board(board, root_player)

        legal = self._legal_moves(board)
        if not legal:
            return self._evaluate_board(board, root_player)

        # Tactical replies always searched first at every node.
        immediate = self._find_immediate_win(board, turn, legal)
        if immediate is not None:
            candidates = [immediate]
        else:
            opp_win = self._find_immediate_win(board, self._other(turn), legal)
            if opp_win is not None:
                candidates = [opp_win]
            else:
                candidates = self._ordered_candidates(board, turn, root_player, self.NODE_CANDIDATE_LIMIT)

        if not candidates:
            return self._evaluate_board(board, root_player)

        if turn == root_player:
            value = -INF
            for move in candidates:
                self._check_time(deadline)
                if board[move[0]][move[1]] != EMPTY:
                    continue
                self._play(board, move, turn)
                try:
                    value = max(
                        value,
                        self._alpha_beta(board, self._other(turn), root_player, depth - 1, alpha, beta, deadline),
                    )
                finally:
                    self._undo(board, move)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = INF
        for move in candidates:
            self._check_time(deadline)
            if board[move[0]][move[1]] != EMPTY:
                continue
            self._play(board, move, turn)
            try:
                value = min(
                    value,
                    self._alpha_beta(board, self._other(turn), root_player, depth - 1, alpha, beta, deadline),
                )
            finally:
                self._undo(board, move)
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _ordered_candidates(
        self,
        board: list[list[int]],
        turn: int,
        root_player: int,
        limit: int,
    ) -> list[tuple[int, int]]:
        """Generate search candidates from both players' shortest paths plus local moves."""
        legal = self._legal_moves(board)
        if not legal:
            return []

        opponent = self._other(turn)
        _turn_dist, turn_path = self._distance_and_path(board, turn)
        _opp_dist, opp_path = self._distance_and_path(board, opponent)

        candidate_set: set[tuple[int, int]] = set()
        for path in (turn_path, opp_path):
            for cell in path:
                if board[cell[0]][cell[1]] == EMPTY:
                    candidate_set.add(cell)
                for neigh in self._neighbors[cell]:
                    if board[neigh[0]][neigh[1]] == EMPTY:
                        candidate_set.add(neigh)

        # Add some central/local candidates for diversity so we do not overfit to one
        # extracted shortest path.
        local_sorted = sorted(legal, key=lambda mv: self._static_move_score(board, mv, turn), reverse=True)
        for mv in local_sorted[: self.EXTRA_LOCAL_CANDIDATES]:
            candidate_set.add(mv)

        if not candidate_set:
            candidate_set.update(local_sorted[:limit])

        scored: list[tuple[tuple[int, int], float]] = []
        for mv in candidate_set:
            if board[mv[0]][mv[1]] != EMPTY:
                continue
            score = self._node_move_score(board, mv, turn, root_player)
            scored.append((mv, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [mv for mv, _score in scored[:limit]]

    def _node_move_score(self, board: list[list[int]], move: tuple[int, int], turn: int, root_player: int) -> float:
        """Cheap-ish ordering score for alpha-beta nodes."""
        opp = self._other(turn)
        before_turn, _ = self._distance_and_path(board, turn)
        before_opp, _ = self._distance_and_path(board, opp)
        self._play(board, move, turn)
        try:
            after_turn, _ = self._distance_and_path(board, turn)
            after_opp, _ = self._distance_and_path(board, opp)
        finally:
            self._undo(board, move)

        score = 12.0 * self._bounded_delta(before_turn, after_turn)
        score += 8.0 * self._bounded_delta(after_opp, before_opp)
        score += self._static_move_score(board, move, turn)

        # For opponent nodes, still order by the opponent's best-looking moves. The
        # minimizer in alpha-beta will handle adversarial choice correctly.
        if turn != root_player:
            score += 2.0 * self._fast_local_score(board, move, turn)
        return score

    def _evaluate_board(self, board: list[list[int]], root_player: int) -> float:
        opponent = self._other(root_player)
        root_dist, root_path = self._distance_and_path(board, root_player)
        opp_dist, opp_path = self._distance_and_path(board, opponent)

        if root_dist == 0:
            return 1_000_000
        if opp_dist == 0:
            return -1_000_000

        # Main race score.
        score = 120.0 * (min(opp_dist, 50.0) - min(root_dist, 50.0))

        # Smaller path is better; having many alternate adjacent cells is also useful.
        score += 2.0 * (len(opp_path) - len(root_path))

        # Connection material/local structure.
        root_local = 0.0
        opp_local = 0.0
        for r, row in enumerate(board):
            for c, value in enumerate(row):
                if value == root_player:
                    root_local += self._stone_structure_score(board, (r, c), root_player)
                elif value == opponent:
                    opp_local += self._stone_structure_score(board, (r, c), opponent)
        score += 0.35 * (root_local - opp_local)
        return score

    def _stone_structure_score(self, board: list[list[int]], stone: tuple[int, int], player: int) -> float:
        own = 0
        for nr, nc in self._neighbors[stone]:
            if board[nr][nc] == player:
                own += 1
        return float(own)

    def _check_time(self, deadline: float) -> None:
        if time.perf_counter() >= deadline - 0.06:
            raise SearchTimeout()


if __name__ == "__main__":
    agent = HexAgent()
    agent.start()