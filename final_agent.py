from collections import deque
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

# Transposition table flags.
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2


class SearchTimeout(Exception):
    """Raised internally when the search reaches the wall-clock deadline."""


@dataclass(slots=True)
class TTEntry:
    """Alpha-beta transposition table entry."""

    depth: int
    score: float
    flag: int
    best_move: int | None


class HexAgent(Agent):

    SOFT_TIME_LIMIT_SECONDS = 4.30
    HARD_TIME_LIMIT_SECONDS = 4.78

    # Search parameters. The transposition table and 1D path cache should allow
    # this version to complete depth 4 more often than the previous depth-3 agent.
    ROOT_CANDIDATE_LIMIT = 30
    NODE_CANDIDATE_LIMIT = 13
    EXTRA_LOCAL_CANDIDATES = 10
    MAX_SEARCH_DEPTH = 8
    TWO_PLY_REPLY_LIMIT = 18

    # check periodically inside search.
    CHECK_INTERVAL = 128
    PVS_EPSILON = 0.01

    # Set to False before final submission 
    DEBUG = False

    NEIGHBOR_DIRS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
    CYCLIC_DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

    @override
    def initialize(self, init_data: dict) -> None:
        self.player_id = int(init_data["player_id"])
        self.opponent_id = 1 - self.player_id

        assert self._goal_axis(PLAYER_0) == "horizontal"
        assert self._goal_axis(PLAYER_1) == "vertical"

        self.rng = random.Random(1_431_993 + 97_531 * self.player_id)

        self._cached_size: int | None = None
        self._n2 = 0
        self._neighbors: list[list[int]] = []
        self._row: list[int] = []
        self._col: list[int] = []
        self._bridge_patterns_by_cell: list[list[tuple[int, int, int]]] = []
        self._center_score: list[float] = []
        self._bit: list[int] = []
        self._all_bits_mask = 0
        self._left_edge_mask = 0
        self._right_edge_mask = 0
        self._top_edge_mask = 0
        self._bottom_edge_mask = 0
        self._not_left_edge_mask = 0
        self._not_right_edge_mask = 0

        # Reused Dijkstra/BFS buffers.
        self._dist: list[float] = []
        self._parent: list[int] = []
        self._visited: list[int] = []
        self._visit_mark = 0

        # Zobrist tables are regenerated per board size.
        self._zobrist: list[list[int]] = []

        # Per-move caches; reset in get_move().
        self._distance_cache: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
        self._tt: dict[tuple[int, int], TTEntry] = {}
        self._nodes = 0
        self._killer_moves: list[list[int | None]] = []

    @override
    def get_move(self, state: State) -> Move:
        """Return a legal move for the current state."""
        start = time.perf_counter()
        soft_deadline = start + self.SOFT_TIME_LIMIT_SECONDS
        hard_deadline = start + self.HARD_TIME_LIMIT_SECONDS
        deadline = soft_deadline

        n = int(state.board_size)
        self._ensure_board_tools(n)

        board = self._flatten_board(state.board)
        hash_key = self._hash_board(board)
        bits = self._bits_from_board(board)
        self._distance_cache.clear()
        self._tt.clear()
        self._nodes = 0
        self._killer_moves = [[None, None] for _ in range(self.MAX_SEARCH_DEPTH + 2)]

        player = self.player_id
        if int(state.turn) != player:
            # Keep the returned Move legal if local/dev runner ever calls oddly.
            player = int(state.turn)
        opponent = self._other(player)

        legal_moves = self._legal_moves(board)
        if not legal_moves:
            raise ValueError("No valid moves available.")

        opening = self._opening_move(board, player, legal_moves)
        if opening is not None:
            return self._make_move(player, opening)

        # Full-board tactical safety layer. Never prune these scans.
        winning_move = self._find_immediate_win(board, hash_key, bits, player, legal_moves)
        if winning_move is not None:
            return self._make_move(player, winning_move)

        blocking_move = self._find_immediate_win(board, hash_key, bits, opponent, legal_moves)
        if blocking_move is not None:
            return self._make_move(player, blocking_move)

        ranked = self._rank_root_moves(board, hash_key, bits, player, legal_moves, deadline)
        best_fallback = ranked[0][0] if ranked else legal_moves[0]

        if time.perf_counter() >= deadline - 0.08:
            return self._make_move(player, best_fallback)

        root_candidates = [move for move, _score in ranked[: self.ROOT_CANDIDATE_LIMIT]]
        best_move = best_fallback
        best_score = -INF
        completed_depth = 0
        previous_move: int | None = None
        previous_score: float | None = None
        extended_time = False

        try:
            for depth in range(1, self.MAX_SEARCH_DEPTH + 1):
                if time.perf_counter() >= deadline - 0.08:
                    break
                move, score = self._alpha_beta_root(board, hash_key, bits, player, root_candidates, depth, deadline)

                # Dynamic time extension: if a deeper iteration changes the principal
                # move or reveals a large score swing, spend part of the safety buffer
                # to resolve the tactical instability instead of blindly returning.
                if depth >= 3 and deadline < hard_deadline:
                    move_changed = previous_move is not None and move is not None and move != previous_move
                    score_swung = previous_score is not None and abs(score - previous_score) > 260.0
                    if move_changed or score_swung:
                        deadline = hard_deadline
                        extended_time = True

                if move is not None:
                    best_move = move
                    best_score = score
                    completed_depth = depth
                    previous_move = move
                    previous_score = score
                if abs(score) >= 900_000:
                    break
        except SearchTimeout:
            pass

        if self.DEBUG:
            elapsed = time.perf_counter() - start
            print(
                f"Move decision took {elapsed:.2f}s | legal={len(legal_moves)} "
                f"ranked={len(ranked)} depth={completed_depth} nodes={self._nodes} "
                f"tt={len(self._tt)} dcache={len(self._distance_cache)} "
                f"extended={extended_time} score={best_score:.2f} move={self._idx_to_pos(best_move)}"
            )

        return self._make_move(player, best_move)

    # ------------------------------------------------------------------
    # Setup and representation helpers
    # ------------------------------------------------------------------

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

        # Precompute bridge triples for every cell:
        # (endpoint, carrier_a, carrier_b)
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
        self._visited = [0] * self._n2
        self._visit_mark = 0

        zrng = random.Random(8_675_309 + 31_337 * n)
        self._zobrist = [
            [zrng.getrandbits(64) for _ in range(self._n2)],
            [zrng.getrandbits(64) for _ in range(self._n2)],
        ]

        self._bit = [1 << idx for idx in range(self._n2)]
        self._all_bits_mask = (1 << self._n2) - 1
        self._left_edge_mask = 0
        self._right_edge_mask = 0
        self._top_edge_mask = 0
        self._bottom_edge_mask = 0
        for r in range(n):
            self._left_edge_mask |= self._bit[r * n]
            self._right_edge_mask |= self._bit[r * n + (n - 1)]
        for c in range(n):
            self._top_edge_mask |= self._bit[c]
            self._bottom_edge_mask |= self._bit[(n - 1) * n + c]
        self._not_left_edge_mask = self._all_bits_mask ^ self._left_edge_mask
        self._not_right_edge_mask = self._all_bits_mask ^ self._right_edge_mask

    @staticmethod
    def _goal_axis(player: int) -> str:
        return "horizontal" if player == PLAYER_0 else "vertical"

    @staticmethod
    def _other(player: int) -> int:
        return 1 - player

    def _flatten_board(self, board_2d: list[list[int]]) -> list[int]:
        return [cell for row in board_2d for cell in row]

    def _hash_board(self, board: list[int]) -> int:
        h = 0
        for idx, value in enumerate(board):
            if value == PLAYER_0:
                h ^= self._zobrist[PLAYER_0][idx]
            elif value == PLAYER_1:
                h ^= self._zobrist[PLAYER_1][idx]
        return h

    def _bits_from_board(self, board: list[int]) -> tuple[int, int]:
        p0_bits = 0
        p1_bits = 0
        for idx, value in enumerate(board):
            if value == PLAYER_0:
                p0_bits |= self._bit[idx]
            elif value == PLAYER_1:
                p1_bits |= self._bit[idx]
        return p0_bits, p1_bits

    def _bits_after_play(self, bits: tuple[int, int], idx: int, player: int) -> tuple[int, int]:
        b0, b1 = bits
        bit = self._bit[idx]
        if player == PLAYER_0:
            return b0 | bit, b1
        return b0, b1 | bit

    def _hash_after_play(self, hash_key: int, idx: int, player: int) -> int:
        return hash_key ^ self._zobrist[player][idx]

    def _make_move(self, player: int, idx: int) -> Move:
        r, c = self._idx_to_pos(idx)
        return Move(player=player, position=[r, c])

    def _idx_to_pos(self, idx: int) -> tuple[int, int]:
        return self._row[idx], self._col[idx]

    @staticmethod
    def _play(board: list[int], idx: int, player: int) -> None:
        board[idx] = player

    @staticmethod
    def _undo(board: list[int], idx: int) -> None:
        board[idx] = EMPTY

    def _legal_moves(self, board: list[int]) -> list[int]:
        return [idx for idx, value in enumerate(board) if value == EMPTY]

    @staticmethod
    def _stone_count(board: list[int]) -> int:
        return sum(1 for value in board if value != EMPTY)

    # ------------------------------------------------------------------
    # Opening
    # ------------------------------------------------------------------

    def _opening_move(self, board: list[int], player: int, legal_moves: list[int]) -> int | None:
        n = self._cached_size or 11
        if n % 2 == 0:
            return None
        center = (n // 2) * n + (n // 2)
        stones = self._stone_count(board)
        legal_set = set(legal_moves)

        if stones == 0 and player == PLAYER_0 and center in legal_set:
            return center

        if stones == 1 and player == PLAYER_1:
            cr, cc = n // 2, n // 2
            ring_rc = [
                (cr, cc + 1),
                (cr - 1, cc),
                (cr + 1, cc),
                (cr, cc - 1),
                (cr - 1, cc + 1),
                (cr + 1, cc - 1),
            ]
            choices = [r * n + c for r, c in ring_rc if 0 <= r < n and 0 <= c < n and r * n + c in legal_set]
            if choices:
                choices.sort(key=lambda mv: self._static_move_score(board, mv, player), reverse=True)
                return choices[0]
            if center in legal_set:
                return center

        return None

    # ------------------------------------------------------------------
    # Win checks and full-board immediate tactics
    # ------------------------------------------------------------------

    def _has_bits_won(self, bits: tuple[int, int], player: int) -> bool:
        """Bitboard flood-fill win test for the verified Hex edge mapping."""
        n = self._cached_size or 11
        player_bits = bits[player]
        if player == PLAYER_0:
            seen = player_bits & self._left_edge_mask
            target = self._right_edge_mask
        else:
            seen = player_bits & self._top_edge_mask
            target = self._bottom_edge_mask

        if not seen:
            return False

        frontier = seen
        while frontier:
            expanded = 0
            expanded |= frontier >> n
            expanded |= (frontier << n) & self._all_bits_mask
            expanded |= (frontier & self._not_right_edge_mask) << 1
            expanded |= (frontier & self._not_left_edge_mask) >> 1
            expanded |= (frontier & self._not_right_edge_mask) >> (n - 1)
            expanded |= ((frontier & self._not_left_edge_mask) << (n - 1)) & self._all_bits_mask
            new_frontier = expanded & player_bits & ~seen
            if not new_frontier:
                break
            seen |= new_frontier
            if seen & target:
                return True
            frontier = new_frontier
        return bool(seen & target)

    def _has_player_won(self, board: list[int], player: int) -> bool:
        """Compatibility wrapper; hot search code uses _has_bits_won directly."""
        return self._has_bits_won(self._bits_from_board(board), player)

    def _find_immediate_win(self, board: list[int], hash_key: int, bits: tuple[int, int], player: int, legal_moves: list[int]) -> int | None:
        """Check every legal cell for a one-move win. This scan is never pruned."""
        ordered = sorted(legal_moves, key=lambda mv: self._fast_local_score(board, mv, player), reverse=True)
        own_bits = bits[player]
        for move in ordered:
            if self._has_bits_won(self._replace_player_bits(bits, player, own_bits | self._bit[move]), player):
                return move
        return None

    def _replace_player_bits(self, bits: tuple[int, int], player: int, player_bits: int) -> tuple[int, int]:
        if player == PLAYER_0:
            return player_bits, bits[PLAYER_1]
        return bits[PLAYER_0], player_bits

    def _count_immediate_wins(self, board: list[int], bits: tuple[int, int], player: int, legal_moves: list[int] | None = None, cap: int = 2) -> int:
        if legal_moves is None:
            legal_moves = self._legal_moves(board)
        count = 0
        own_bits = bits[player]
        for move in legal_moves:
            if board[move] != EMPTY:
                continue
            won = self._has_bits_won(self._replace_player_bits(bits, player, own_bits | self._bit[move]), player)
            if won:
                count += 1
                if count >= cap:
                    return count
        return count

    # ------------------------------------------------------------------
    # Dijkstra distance/path cache
    # ------------------------------------------------------------------

    def _distance_and_path(self, board: list[int], hash_key: int, player: int) -> tuple[float, tuple[int, ...]]:
        cached = self._distance_cache.get((hash_key, player))
        if cached is not None:
            return cached

        n = self._cached_size or 11
        opponent = self._other(player)
        dist = self._dist
        parent = self._parent

        # Reset reused buffers. For 121 cells this is much cheaper than building 2D arrays.
        for i in range(self._n2):
            dist[i] = float(INF)
            parent[i] = -1

        heap: list[tuple[float, int]] = []

        def cell_cost(idx: int) -> float:
            value = board[idx]
            if value == player:
                return 0.0
            if value == EMPTY:
                return 1.0
            if value == opponent:
                return float(INF)
            return float(INF)

        if player == PLAYER_0:
            starts = [r * n for r in range(n)]
        else:
            starts = [c for c in range(n)]

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

            # Normal adjacent graph edges.
            for nxt in self._neighbors[cur]:
                step = cell_cost(nxt)
                if step >= INF:
                    continue
                nd = cur_dist + step
                if nd < dist[nxt]:
                    dist[nxt] = nd
                    parent[nxt] = cur
                    heapq.heappush(heap, (nd, nxt))

            # Cheap virtual connection for already-formed bridges between own stones.
            # This is not a full electrical solver, but it makes bridges visible to
            # the race distance without expensive hard-coded root simulations.
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
            if board[cur] == EMPTY:
                path.append(cur)
            cur = parent[cur]
        result = (best_cost, tuple(path))
        self._distance_cache[(hash_key, player)] = result
        return result

    # ------------------------------------------------------------------
    # Root ranking and danger guard
    # ------------------------------------------------------------------

    def _rank_root_moves(
        self,
        board: list[int],
        hash_key: int,
        bits: tuple[int, int],
        player: int,
        legal_moves: list[int],
        deadline: float,
    ) -> list[tuple[int, float]]:
        opponent = self._other(player)
        my_before, my_path = self._distance_and_path(board, hash_key, player)
        opp_before, opp_path = self._distance_and_path(board, hash_key, opponent)
        stones = self._stone_count(board)
        my_path_set = set(my_path)
        opp_path_set = set(opp_path)

        ranked: list[tuple[int, float]] = []
        for move in legal_moves:
            # Root ranking performs heavy work; check time explicitly here.
            if time.perf_counter() >= deadline - 0.12:
                break

            new_hash = self._hash_after_play(hash_key, move, player)
            board[move] = player
            my_after, _ = self._distance_and_path(board, new_hash, player)
            opp_after, _ = self._distance_and_path(board, new_hash, opponent)
            board[move] = EMPTY

            my_gain = self._bounded_delta(my_before, my_after)
            opp_disruption = self._bounded_delta(opp_after, opp_before)
            disruption_weight = 13.5 if player == PLAYER_1 else 10.5

            score = 17.0 * my_gain + disruption_weight * opp_disruption
            score += self._static_move_score(board, move, player)

            if move in my_path_set:
                score += 7.0
            if move in opp_path_set:
                score += 9.0
            if self._is_neighbor_of_set(move, my_path_set):
                score += 2.2
            if self._is_neighbor_of_set(move, opp_path_set):
                score += 3.4

            r, c = self._row[move], self._col[move]
            n = self._cached_size or 11
            if stones < 18 and (r == 0 or c == 0 or r == n - 1 or c == n - 1):
                score -= 8.0
            if stones < 24 and (r, c) in {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}:
                score -= 5.0

            score -= self._root_danger_score_after_move(board, hash_key, bits, move, player, deadline)
            ranked.append((move, score))

        if not ranked:
            ranked = [(move, self._static_move_score(board, move, player)) for move in legal_moves]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    @staticmethod
    def _bounded_delta(before: float, after: float) -> float:
        return min(before, 50.0) - min(after, 50.0)

    def _is_neighbor_of_set(self, move: int, targets: set[int]) -> bool:
        return bool(targets) and any(nbr in targets for nbr in self._neighbors[move])

    def _root_danger_score_after_move(self, board: list[int], hash_key: int, bits: tuple[int, int], move: int, player: int, deadline: float) -> float:
        """Penalize moves that allow opponent immediate wins or next-move forks."""
        opponent = self._other(player)
        new_hash = self._hash_after_play(hash_key, move, player)
        new_bits = self._bits_after_play(bits, move, player)
        board[move] = player
        try:
            remaining = self._legal_moves(board)
            immediate = self._count_immediate_wins(board, new_bits, opponent, remaining, cap=2)
            if immediate >= 2:
                return 600_000.0
            if immediate == 1:
                return 250_000.0

            opp_dist, opp_path = self._distance_and_path(board, new_hash, opponent)
            if opp_dist > 3 and len(opp_path) > 4:
                return 0.0

            replies = self._ordered_candidates(board, new_hash, opponent, opponent, self.TWO_PLY_REPLY_LIMIT)
            for cell in opp_path:
                if board[cell] == EMPTY and cell not in replies:
                    replies.append(cell)

            worst = 0.0
            limit = max(self.TWO_PLY_REPLY_LIMIT, len(opp_path))
            for reply in replies[:limit]:
                if time.perf_counter() >= deadline - 0.10:
                    break
                if board[reply] != EMPTY:
                    continue
                reply_hash = self._hash_after_play(new_hash, reply, opponent)
                reply_bits = self._bits_after_play(new_bits, reply, opponent)
                board[reply] = opponent
                try:
                    if self._has_bits_won(reply_bits, opponent):
                        return 500_000.0
                    after_legal = self._legal_moves(board)
                    fork_count = self._count_immediate_wins(board, reply_bits, opponent, after_legal, cap=2)
                    if fork_count >= 2:
                        return 300_000.0
                    if fork_count == 1:
                        worst = max(worst, 5_000.0)
                    # Keep the hash used so linters do not complain in local checks.
                    _ = reply_hash
                finally:
                    board[reply] = EMPTY
            return worst
        finally:
            board[move] = EMPTY

    # ------------------------------------------------------------------
    # Alpha-beta with transposition table
    # ------------------------------------------------------------------

    def _order_with_tt_and_killers(
        self,
        candidates: list[int],
        tt_move: int | None,
        depth: int,
    ) -> list[int]:
        """Order moves as TT move, killer moves at same remaining depth, then heuristic order."""
        ordered = candidates[:]
        priority: list[int] = []
        if tt_move is not None:
            priority.append(tt_move)
        if 0 <= depth < len(self._killer_moves):
            for killer in self._killer_moves[depth]:
                if killer is not None:
                    priority.append(killer)

        for mv in reversed(priority):
            if mv in ordered:
                ordered.remove(mv)
                ordered.insert(0, mv)
        return ordered

    def _store_killer(self, depth: int, move: int) -> None:
        if not (0 <= depth < len(self._killer_moves)):
            return
        killers = self._killer_moves[depth]
        if killers[0] == move:
            return
        killers[1] = killers[0]
        killers[0] = move

    def _alpha_beta_root(
        self,
        board: list[int],
        hash_key: int,
        bits: tuple[int, int],
        root_player: int,
        candidates: list[int],
        depth: int,
        deadline: float,
    ) -> tuple[int | None, float]:
        alpha = -float(INF)
        beta = float(INF)
        best_move: int | None = None
        best_score = -float(INF)

        # Hash move ordering from previous iterations via TT best move if available.
        root_key = (hash_key, root_player)
        tt_move = self._tt.get(root_key).best_move if self._tt.get(root_key) else None
        ordered = self._order_with_tt_and_killers(candidates, tt_move, depth)

        first_child = True
        for move in ordered:
            self._check_time(deadline, force=True)
            if board[move] != EMPTY:
                continue
            child_hash = self._hash_after_play(hash_key, move, root_player)
            child_bits = self._bits_after_play(bits, move, root_player)
            board[move] = root_player
            try:
                if self._has_bits_won(child_bits, root_player):
                    score = 1_000_000.0 + depth
                else:
                    if first_child:
                        score = self._alpha_beta(
                            board,
                            child_hash,
                            child_bits,
                            turn=self._other(root_player),
                            root_player=root_player,
                            depth=depth - 1,
                            alpha=alpha,
                            beta=beta,
                            deadline=deadline,
                        )
                    else:
                        score = self._alpha_beta(
                            board,
                            child_hash,
                            child_bits,
                            turn=self._other(root_player),
                            root_player=root_player,
                            depth=depth - 1,
                            alpha=alpha,
                            beta=alpha + self.PVS_EPSILON,
                            deadline=deadline,
                        )
                        if alpha < score < beta:
                            score = self._alpha_beta(
                                board,
                                child_hash,
                                child_bits,
                                turn=self._other(root_player),
                                root_player=root_player,
                                depth=depth - 1,
                                alpha=alpha,
                                beta=beta,
                                deadline=deadline,
                            )
            finally:
                board[move] = EMPTY

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            first_child = False

        return best_move, best_score

    def _alpha_beta(
        self,
        board: list[int],
        hash_key: int,
        bits: tuple[int, int],
        turn: int,
        root_player: int,
        depth: int,
        alpha: float,
        beta: float,
        deadline: float,
    ) -> float:
        self._nodes += 1
        self._check_time(deadline)

        alpha_orig = alpha
        beta_orig = beta
        key = (hash_key, turn)
        entry = self._tt.get(key)
        if entry is not None and entry.depth >= depth:
            if entry.flag == TT_EXACT:
                return entry.score
            if entry.flag == TT_LOWER:
                alpha = max(alpha, entry.score)
            elif entry.flag == TT_UPPER:
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score

        opponent = self._other(root_player)
        if self._has_bits_won(bits, root_player):
            return 1_000_000.0 + depth
        if self._has_bits_won(bits, opponent):
            return -1_000_000.0 - depth
        if depth <= 0:
            return self._evaluate_board(board, hash_key, root_player)

        candidates = self._ordered_candidates(board, hash_key, turn, root_player, self.NODE_CANDIDATE_LIMIT)
        if not candidates:
            return self._evaluate_board(board, hash_key, root_player)

        tt_best = entry.best_move if entry is not None else None
        candidates = self._order_with_tt_and_killers(candidates, tt_best, depth)

        best_move: int | None = None
        first_child = True
        if turn == root_player:
            value = -float(INF)
            for move in candidates:
                if board[move] != EMPTY:
                    continue
                child_hash = self._hash_after_play(hash_key, move, turn)
                child_bits = self._bits_after_play(bits, move, turn)
                board[move] = turn
                try:
                    if first_child:
                        child_score = self._alpha_beta(board, child_hash, child_bits, self._other(turn), root_player, depth - 1, alpha, beta, deadline)
                    else:
                        child_score = self._alpha_beta(board, child_hash, child_bits, self._other(turn), root_player, depth - 1, alpha, alpha + self.PVS_EPSILON, deadline)
                        if alpha < child_score < beta:
                            child_score = self._alpha_beta(board, child_hash, child_bits, self._other(turn), root_player, depth - 1, alpha, beta, deadline)
                finally:
                    board[move] = EMPTY
                if child_score > value:
                    value = child_score
                    best_move = move
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    self._store_killer(depth, move)
                    break
                first_child = False
        else:
            value = float(INF)
            for move in candidates:
                if board[move] != EMPTY:
                    continue
                child_hash = self._hash_after_play(hash_key, move, turn)
                child_bits = self._bits_after_play(bits, move, turn)
                board[move] = turn
                try:
                    if first_child:
                        child_score = self._alpha_beta(board, child_hash, child_bits, self._other(turn), root_player, depth - 1, alpha, beta, deadline)
                    else:
                        child_score = self._alpha_beta(board, child_hash, child_bits, self._other(turn), root_player, depth - 1, beta - self.PVS_EPSILON, beta, deadline)
                        if alpha < child_score < beta:
                            child_score = self._alpha_beta(board, child_hash, child_bits, self._other(turn), root_player, depth - 1, alpha, beta, deadline)
                finally:
                    board[move] = EMPTY
                if child_score < value:
                    value = child_score
                    best_move = move
                if value < beta:
                    beta = value
                if alpha >= beta:
                    self._store_killer(depth, move)
                    break
                first_child = False

        if value <= alpha_orig:
            flag = TT_UPPER
        elif value >= beta_orig:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        self._tt[key] = TTEntry(depth=depth, score=value, flag=flag, best_move=best_move)
        return value

    def _ordered_candidates(self, board: list[int], hash_key: int, turn: int, root_player: int, limit: int) -> list[int]:
        legal = self._legal_moves(board)
        if not legal:
            return []

        opponent = self._other(turn)
        _turn_dist, turn_path = self._distance_and_path(board, hash_key, turn)
        _opp_dist, opp_path = self._distance_and_path(board, hash_key, opponent)

        candidate_set: set[int] = set()
        for path in (turn_path, opp_path):
            for cell in path:
                if board[cell] == EMPTY:
                    candidate_set.add(cell)
                for nbr in self._neighbors[cell]:
                    if board[nbr] == EMPTY:
                        candidate_set.add(nbr)

        local_sorted = sorted(legal, key=lambda mv: self._static_move_score(board, mv, turn), reverse=True)
        for mv in local_sorted[: self.EXTRA_LOCAL_CANDIDATES]:
            candidate_set.add(mv)

        if not candidate_set:
            candidate_set.update(local_sorted[:limit])

        turn_path_set = set(turn_path)
        opp_path_set = set(opp_path)
        scored: list[tuple[int, float]] = []
        for mv in candidate_set:
            if board[mv] != EMPTY:
                continue
            score = self._fast_candidate_score(board, mv, turn, turn_path_set, opp_path_set)
            scored.append((mv, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [mv for mv, _score in scored[:limit]]

    # ------------------------------------------------------------------
    # Evaluation and static features
    # ------------------------------------------------------------------

    def _evaluate_board(self, board: list[int], hash_key: int, root_player: int) -> float:
        opponent = self._other(root_player)
        root_dist, root_path = self._distance_and_path(board, hash_key, root_player)
        opp_dist, opp_path = self._distance_and_path(board, hash_key, opponent)

        if root_dist == 0:
            return 1_000_000.0
        if opp_dist == 0:
            return -1_000_000.0

        score = 120.0 * (min(opp_dist, 50.0) - min(root_dist, 50.0))
        score += 2.0 * (len(opp_path) - len(root_path))

        # Cheap structure difference. Avoid calling expensive functions here.
        root_local = 0.0
        opp_local = 0.0
        for idx, value in enumerate(board):
            if value == root_player:
                root_local += self._stone_structure_score(board, idx, root_player)
            elif value == opponent:
                opp_local += self._stone_structure_score(board, idx, opponent)
        score += 0.35 * (root_local - opp_local)
        return score

    def _fast_candidate_score(self, board: list[int], move: int, player: int, my_path: set[int], opp_path: set[int]) -> float:
        score = self._static_move_score(board, move, player)
        if move in my_path:
            score += 7.0
        if move in opp_path:
            score += 9.0
        if self._is_neighbor_of_set(move, my_path):
            score += 2.0
        if self._is_neighbor_of_set(move, opp_path):
            score += 3.0
        return score

    def _static_move_score(self, board: list[int], move: int, player: int) -> float:
        score = self._fast_local_score(board, move, player)
        score += 2.6 * self._bridge_creation_count(board, move, player)
        score += 2.3 * self._opponent_bridge_attack_count(board, move, self._other(player))
        score += self._edge_orientation_bonus(move, player)
        return score

    def _fast_local_score(self, board: list[int], move: int, player: int) -> float:
        opponent = self._other(player)
        own_neighbors = 0
        opp_neighbors = 0
        empty_neighbors = 0
        for nbr in self._neighbors[move]:
            value = board[nbr]
            if value == player:
                own_neighbors += 1
            elif value == opponent:
                opp_neighbors += 1
            else:
                empty_neighbors += 1
        return 1.05 * own_neighbors + 0.62 * opp_neighbors + 0.09 * empty_neighbors + self._center_score[move]

    def _edge_orientation_bonus(self, move: int, player: int) -> float:
        n = self._cached_size or 11
        nearest = min(self._col[move], n - 1 - self._col[move]) if player == PLAYER_0 else min(self._row[move], n - 1 - self._row[move])
        return 0.08 * (n / 2.0 - nearest)

    def _bridge_creation_count(self, board: list[int], move: int, player: int) -> int:
        opponent = self._other(player)
        count = 0
        for endpoint, carrier_a, carrier_b in self._bridge_patterns_by_cell[move]:
            if board[endpoint] == player and board[carrier_a] != opponent and board[carrier_b] != opponent:
                count += 1
        return count

    def _opponent_bridge_attack_count(self, board: list[int], move: int, opponent: int) -> int:
        count = 0
        seen: set[tuple[int, int]] = set()
        for adjacent in self._neighbors[move]:
            if board[adjacent] != opponent:
                continue
            for endpoint, carrier_a, carrier_b in self._bridge_patterns_by_cell[adjacent]:
                if board[endpoint] != opponent:
                    continue
                if move != carrier_a and move != carrier_b:
                    continue
                other_carrier = carrier_b if move == carrier_a else carrier_a
                if board[other_carrier] == EMPTY:
                    endpoints = (adjacent, endpoint) if adjacent < endpoint else (endpoint, adjacent)
                    if endpoints not in seen:
                        seen.add(endpoints)
                        count += 1
        return count

    def _stone_structure_score(self, board: list[int], stone: int, player: int) -> float:
        own = 0
        for nbr in self._neighbors[stone]:
            if board[nbr] == player:
                own += 1
        return float(own)

    def _check_time(self, deadline: float, force: bool = False) -> None:
        if force or (self._nodes & (self.CHECK_INTERVAL - 1)) == 0:
            if time.perf_counter() >= deadline - 0.06:
                raise SearchTimeout()


if __name__ == "__main__":
    agent = HexAgent()
    agent.start()