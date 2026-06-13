"""
Competitive Hex agent for the AICA Game AI Platform.

Design goals
------------
1. Always return a legal move well before the 5-second platform timeout.
2. Verify and use the real gamelib edge mapping:
      Player 0 connects LEFT  -> RIGHT.
      Player 1 connects TOP   -> BOTTOM.
3. Never prune immediate tactical checks: win-in-1 and block-loss-in-1 scan all empty cells.
4. Use a fast Hex-aware evaluator for move ordering.
5. Use time-managed MCTS with lightweight biased rollouts.

Only the Python standard library and gamelib are used.
"""

from collections import deque
import heapq
import math
import random
import time
from dataclasses import dataclass, field
from typing import override

from gamelib.hex.agent import Agent
from gamelib.hex.gamestate import GameState as State
from gamelib.hex.move import Move


# Engine status values copied from gamelib.hex.engine.GameStatus.
ONGOING = -1
PLAYER_0_WINS = 0
PLAYER_1_WINS = 1

# Board encoding copied from the gamelib docs/source.
EMPTY = -1
PLAYER_0 = 0
PLAYER_1 = 1

INF = 10**9


@dataclass(slots=True)
class SearchNode:
    """Small MCTS node storing statistics from the root player's perspective."""

    board: list[list[int]]
    player_to_move: int
    parent: object = None
    move: tuple[int, int] | None = None
    player_just_moved: int | None = None
    prior_score: float = 0.0
    visits: int = 0
    wins: float = 0.0
    children: list = field(default_factory=list)
    untried_moves: list[tuple[int, int]] | None = None


class HexAgent(Agent):
    """
    Hybrid tactical + heuristic + MCTS Hex agent.

    The implementation intentionally keeps the expensive Hex knowledge at the root
    and expansion stages. Rollouts are lightweight so that we can complete enough
    simulations inside the 5-second limit.
    """

    # Platform timeout is 5.0s; keep a large safety buffer.
    MOVE_TIME_LIMIT_SECONDS = 4.20

    # MCTS tuning. These values are intentionally conservative for long Hex games.
    UCT_C = 0.85
    ROOT_CANDIDATE_LIMIT = 42
    TREE_CANDIDATE_LIMIT = 14
    ROLLOUT_SAMPLE_SIZE = 10
    ENDGAME_SOLVE_EMPTY_LIMIT = 8
    ENDGAME_SOLVE_TIME_BUDGET = 0.65

    # Hex neighbor directions used by the provided Engine source.
    NEIGHBOR_DIRS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

    # Cyclic directions make bridge pattern construction easier.
    CYCLIC_DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

    @override
    def initialize(self, init_data: dict) -> None:
        """Store player id and prepare deterministic random state."""
        self.player_id = int(init_data["player_id"])
        self.opponent_id = 1 - self.player_id

        # Confirm the edge mapping discovered from gamelib.hex.engine:
        #   Player 0: left-right, Player 1: top-bottom.
        # These assertions catch accidental future edits to this file.
        assert self._goal_axis(PLAYER_0) == "horizontal"
        assert self._goal_axis(PLAYER_1) == "vertical"

        self.rng = random.Random(827_361 + 104_729 * self.player_id)
        self._cached_size: int | None = None
        self._neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._bridge_patterns: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []

    @override
    def get_move(self, state: State) -> Move:
        """Return a legal move for the current position."""
        start_time = time.perf_counter()
        deadline = start_time + self.MOVE_TIME_LIMIT_SECONDS

        board = [row[:] for row in state.board]
        n = int(state.board_size)
        self._ensure_board_tools(n)

        legal_moves = self._legal_moves(board)
        if not legal_moves:
            # This should not happen in a normal Hex game, but avoid returning nonsense.
            raise ValueError("No valid moves available.")

        # If the platform ever calls us when it is not our turn, still return a legal
        # Move object for state.turn rather than producing an invalid player id.
        player = int(state.turn)
        if player != self.player_id:
            player = self.player_id

        # Opening book: no swap rule, odd 11x11 board, so the center is the strongest
        # first-player opening and costs zero search time.
        opening_move = self._opening_move(board, player, legal_moves)
        if opening_move is not None:
            return Move(player=player, position=[opening_move[0], opening_move[1]])

        # Unconditional tactical layer. This scans every empty cell before any move
        # ordering or pruning so we never miss a one-move win or immediate block.
        winning_move = self._find_immediate_win(board, player, legal_moves)
        if winning_move is not None:
            return Move(player=player, position=[winning_move[0], winning_move[1]])

        opponent_winner = self._find_immediate_win(board, 1 - player, legal_moves)
        if opponent_winner is not None:
            return Move(player=player, position=[opponent_winner[0], opponent_winner[1]])

        # Deterministic fallback is updated before search begins. If time runs out or
        # MCTS completes very few iterations, this is still a sensible legal move.
        ranked_moves = self._rank_root_moves(board, player, legal_moves)
        best_fallback = ranked_moves[0][0]

        # Small exact-ish endgame solver. It is deliberately time-capped and only used
        # for tiny positions; otherwise MCTS remains the main search method.
        if len(legal_moves) <= self.ENDGAME_SOLVE_EMPTY_LIMIT:
            endgame_deadline = min(deadline, time.perf_counter() + self.ENDGAME_SOLVE_TIME_BUDGET)
            solved_move = self._try_endgame_solve(board, player, legal_moves, endgame_deadline)
            if solved_move is not None:
                return Move(player=player, position=[solved_move[0], solved_move[1]])

        if time.perf_counter() >= deadline - 0.05:
            return Move(player=player, position=[best_fallback[0], best_fallback[1]])

        mcts_move = self._run_mcts(board, player, ranked_moves, deadline)
        if mcts_move is None:
            mcts_move = best_fallback

        end_time = time.perf_counter()
        diff_time = end_time - start_time
        print(f"Move decision took {diff_time:.2f}s with {len(legal_moves)} legal moves and {len(ranked_moves)} ranked moves.")

        return Move(player=player, position=[mcts_move[0], mcts_move[1]])

    # -------------------------------------------------------------------------
    # Board setup and low-level helpers
    # -------------------------------------------------------------------------

    def _ensure_board_tools(self, n: int) -> None:
        """Cache neighbors and bridge geometry for a board size."""
        if self._cached_size == n:
            return

        self._cached_size = n
        self._neighbors = {}
        for r in range(n):
            for c in range(n):
                neigh = []
                for dr, dc in self.NEIGHBOR_DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        neigh.append((nr, nc))
                self._neighbors[(r, c)] = neigh

        # A bridge endpoint is two hex-steps away through two adjacent carrier cells.
        # Pattern tuple: (endpoint_offset, carrier_offset_1, carrier_offset_2)
        patterns = []
        for i, first in enumerate(self.CYCLIC_DIRS):
            second = self.CYCLIC_DIRS[(i + 1) % len(self.CYCLIC_DIRS)]
            endpoint = (first[0] + second[0], first[1] + second[1])
            patterns.append((endpoint, first, second))
        self._bridge_patterns = patterns

    @staticmethod
    def _goal_axis(player: int) -> str:
        """Return the verified gamelib edge objective for a player."""
        return "horizontal" if player == PLAYER_0 else "vertical"

    @staticmethod
    def _other(player: int) -> int:
        return 1 - player

    @staticmethod
    def _copy_board(board: list[list[int]]) -> list[list[int]]:
        return [row[:] for row in board]

    @staticmethod
    def _play_in_place(board: list[list[int]], move: tuple[int, int], player: int) -> None:
        board[move[0]][move[1]] = player

    @staticmethod
    def _undo_in_place(board: list[list[int]], move: tuple[int, int]) -> None:
        board[move[0]][move[1]] = EMPTY

    def _legal_moves(self, board: list[list[int]]) -> list[tuple[int, int]]:
        return [
            (r, c)
            for r, row in enumerate(board)
            for c, value in enumerate(row)
            if value == EMPTY
        ]

    def _stone_count(self, board: list[list[int]]) -> int:
        return sum(1 for row in board for value in row if value != EMPTY)

    # -------------------------------------------------------------------------
    # Win detection and tactical layer
    # -------------------------------------------------------------------------

    def _has_player_won(self, board: list[list[int]], player: int) -> bool:
        """Fast BFS win check using gamelib's edge mapping."""
        n = len(board)
        visited = [[False] * n for _ in range(n)]
        queue: deque[tuple[int, int]] = deque()

        if player == PLAYER_0:
            # Player 0 connects left to right.
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
            # Player 1 connects top to bottom.
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
        """Scan every legal cell and return a one-move win if one exists."""
        # Use root ranking only as order, never as pruning. This improves tie-breaking
        # while still checking all legal cells.
        for move in self._cheap_tactical_order(board, player, legal_moves):
            self._play_in_place(board, move, player)
            won = self._has_player_won(board, player)
            self._undo_in_place(board, move)
            if won:
                return move
        return None

    def _cheap_tactical_order(
        self,
        board: list[list[int]],
        player: int,
        legal_moves: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Cheap order for full tactical scans. Does not prune."""
        return sorted(
            legal_moves,
            key=lambda mv: self._quick_local_score(board, mv, player),
            reverse=True,
        )

    # -------------------------------------------------------------------------
    # Opening logic
    # -------------------------------------------------------------------------

    def _opening_move(
        self,
        board: list[list[int]],
        player: int,
        legal_moves: list[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """Small no-swap opening book for the first two plies."""
        n = len(board)
        if n % 2 == 0:
            return None

        center = (n // 2, n // 2)
        stones = self._stone_count(board)
        legal_set = set(legal_moves)

        # First player, first move: exact center.
        if stones == 0 and player == PLAYER_0 and center in legal_set:
            return center

        # Second player response to a center or near-center opening. We only choose
        # from a small ring and score defensively; this avoids wasting MCTS time.
        if stones == 1 and player == PLAYER_1:
            response_ring = [
                (center[0] - 1, center[1]),
                (center[0] + 1, center[1]),
                (center[0], center[1] - 1),
                (center[0], center[1] + 1),
                (center[0] - 1, center[1] + 1),
                (center[0] + 1, center[1] - 1),
                (center[0] - 2, center[1] + 1),
                (center[0] + 2, center[1] - 1),
                (center[0] - 1, center[1] - 1),
                (center[0] + 1, center[1] + 1),
            ]
            candidates = [mv for mv in response_ring if mv in legal_set]
            if candidates:
                ranked = self._rank_root_moves(board, player, candidates)
                return ranked[0][0]

        return None

    # -------------------------------------------------------------------------
    # Heuristic evaluation and move ordering
    # -------------------------------------------------------------------------

    def _rank_root_moves(
        self,
        board: list[list[int]],
        player: int,
        legal_moves: list[tuple[int, int]],
    ) -> list[tuple[tuple[int, int], float]]:
        """
        Heavy but still fast root move ranking.

        It uses full weighted connection distances for both players before/after
        every move, plus local Hex features such as bridge creation and bridge attack.
        """
        opponent = self._other(player)
        my_before = self._connection_distance(board, player)
        opp_before = self._connection_distance(board, opponent)
        stones = self._stone_count(board)

        ranked: list[tuple[tuple[int, int], float]] = []
        for move in legal_moves:
            self._play_in_place(board, move, player)
            my_after = self._connection_distance(board, player)
            opp_after = self._connection_distance(board, opponent)
            self._undo_in_place(board, move)

            my_gain = self._bounded_distance_delta(my_before, my_after)
            opp_disruption = self._bounded_distance_delta(opp_after, opp_before)

            # Second player has no swap-rule compensation, so it plays more actively
            # against player 0's connection race.
            disruption_weight = 11.5 if player == PLAYER_1 else 8.5

            score = 15.0 * my_gain + disruption_weight * opp_disruption
            score += self._quick_local_score(board, move, player)
            score += 2.4 * self._bridge_creation_count(board, move, player)
            score += 2.1 * self._opponent_bridge_attack_count(board, move, opponent)
            score += self._edge_orientation_bonus(board, move, player)

            # Do not let the distance metric overvalue naked early edge moves.
            # Tactical edge wins/blocks were already handled before ranking.
            r, c = move
            if stones < 15 and (r == 0 or c == 0 or r == len(board) - 1 or c == len(board) - 1):
                score -= 2.75
            ranked.append((move, score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    @staticmethod
    def _bounded_distance_delta(before: float, after: float) -> float:
        """Keep infinity-like Dijkstra values from dominating the score."""
        before = min(float(before), 50.0)
        after = min(float(after), 50.0)
        return before - after

    def _connection_distance(self, board: list[list[int]], player: int) -> float:
        """
        Dijkstra connection distance.

        Own stones cost 0, empty cells cost 1, opponent stones are impassable. This
        is a fast race estimate. Bridge and template effects are added separately in
        move scoring instead of making every path calculation complex.
        """
        n = len(board)
        opponent = self._other(player)
        dist = [[INF] * n for _ in range(n)]
        heap: list[tuple[int, int, int]] = []

        def cell_cost(r: int, c: int) -> int:
            value = board[r][c]
            if value == player:
                return 0
            if value == EMPTY:
                return 1
            if value == opponent:
                return INF
            return INF

        if player == PLAYER_0:
            for r in range(n):
                cost = cell_cost(r, 0)
                if cost < INF:
                    dist[r][0] = cost
                    heapq.heappush(heap, (cost, r, 0))
        else:
            for c in range(n):
                cost = cell_cost(0, c)
                if cost < INF:
                    dist[0][c] = cost
                    heapq.heappush(heap, (cost, 0, c))

        best = INF
        while heap:
            cur_dist, r, c = heapq.heappop(heap)
            if cur_dist != dist[r][c]:
                continue

            if player == PLAYER_0 and c == n - 1:
                best = cur_dist
                break
            if player == PLAYER_1 and r == n - 1:
                best = cur_dist
                break

            for nr, nc in self._neighbors[(r, c)]:
                step = cell_cost(nr, nc)
                if step >= INF:
                    continue
                nd = cur_dist + step
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))

        return float(best)

    def _quick_local_score(self, board: list[list[int]], move: tuple[int, int], player: int) -> float:
        """Cheap local features usable in tactical scans, tree nodes, and rollouts."""
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

        # Keep center attractive, especially early. Manhattan is good enough here.
        center = (n - 1) / 2.0
        center_distance = abs(r - center) + abs(c - center)
        center_bonus = max(0.0, 7.5 - center_distance) * 0.22

        # Local connectivity and contact with opponent threats.
        score = 0.95 * own_neighbors + 0.48 * opp_neighbors + 0.12 * empty_neighbors + center_bonus

        # Acute corners are usually low-value unless they are tactical. Keep this
        # constant-time because the same function is also used inside rollouts.
        if (r, c) in {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}:
            score -= 0.65

        return score

    def _edge_orientation_bonus(self, board: list[list[int]], move: tuple[int, int], player: int) -> float:
        """Small edge-aware bonus, oriented to the player's actual target sides."""
        n = len(board)
        r, c = move
        if player == PLAYER_0:
            nearest_goal_edge = min(c, n - 1 - c)
        else:
            nearest_goal_edge = min(r, n - 1 - r)

        # Small constant-time edge signal. Direct edge contacts are not always best
        # early, so this remains deliberately weak.
        return 0.06 * (n / 2.0 - nearest_goal_edge)

    def _bridge_creation_count(self, board: list[list[int]], move: tuple[int, int], player: int) -> int:
        """Count simple bridge patterns created by placing player at move."""
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
        """
        Count whether move attacks a carrier cell of an opponent bridge.

        This does not mean every bridge intrusion is good, but it is a useful candidate
        prior and the search can reject bad intrusions later.
        """
        n = len(board)
        mr, mc = move
        count = 0
        seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()

        for ar, ac in self._neighbors[(mr, mc)]:
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
                if not (0 <= orow < n and 0 <= ocol < n):
                    continue
                if board[orow][ocol] == EMPTY:
                    endpoints = tuple(sorted(((ar, ac), (er, ec))))
                    if endpoints not in seen:
                        seen.add(endpoints)
                        count += 1
        return count

    # -------------------------------------------------------------------------
    # Endgame search
    # -------------------------------------------------------------------------

    def _try_endgame_solve(
        self,
        board: list[list[int]],
        player: int,
        legal_moves: list[tuple[int, int]],
        deadline: float,
    ) -> tuple[int, int] | None:
        """Tiny time-capped exact search. Returns a forced winning move if found."""
        ordered = [move for move, _ in self._rank_root_moves(board, player, legal_moves)]
        root = player
        memo: dict[tuple[tuple[tuple[int, ...], ...], int], bool] = {}

        def key_for(cur_board: list[list[int]], turn: int) -> tuple[tuple[tuple[int, ...], ...], int]:
            return tuple(tuple(row) for row in cur_board), turn

        def root_can_force_win(cur_board: list[list[int]], turn: int, empties: list[tuple[int, int]]) -> bool:
            if time.perf_counter() >= deadline:
                raise TimeoutError
            if self._has_player_won(cur_board, root):
                return True
            if self._has_player_won(cur_board, self._other(root)):
                return False
            if not empties:
                return False

            memo_key = key_for(cur_board, turn)
            cached = memo.get(memo_key)
            if cached is not None:
                return cached

            # Move ordering matters a lot even for tiny endgames.
            ordered_empties = sorted(
                empties,
                key=lambda mv: self._quick_local_score(cur_board, mv, turn),
                reverse=True,
            )

            if turn == root:
                for mv in ordered_empties:
                    cur_board[mv[0]][mv[1]] = turn
                    next_empties = [x for x in empties if x != mv]
                    result = root_can_force_win(cur_board, self._other(turn), next_empties)
                    cur_board[mv[0]][mv[1]] = EMPTY
                    if result:
                        memo[memo_key] = True
                        return True
                memo[memo_key] = False
                return False

            # Opponent chooses the move worst for root.
            for mv in ordered_empties:
                cur_board[mv[0]][mv[1]] = turn
                next_empties = [x for x in empties if x != mv]
                result = root_can_force_win(cur_board, self._other(turn), next_empties)
                cur_board[mv[0]][mv[1]] = EMPTY
                if not result:
                    memo[memo_key] = False
                    return False
            memo[memo_key] = True
            return True

        try:
            for move in ordered:
                board[move[0]][move[1]] = player
                if self._has_player_won(board, player):
                    board[move[0]][move[1]] = EMPTY
                    return move
                remaining = [mv for mv in legal_moves if mv != move]
                winning = root_can_force_win(board, self._other(player), remaining)
                board[move[0]][move[1]] = EMPTY
                if winning:
                    return move
        except TimeoutError:
            return None

        return None

    # -------------------------------------------------------------------------
    # MCTS
    # -------------------------------------------------------------------------

    def _run_mcts(
        self,
        board: list[list[int]],
        player: int,
        ranked_moves: list[tuple[tuple[int, int], float]],
        deadline: float,
    ) -> tuple[int, int] | None:
        """Run wall-clock limited MCTS and return the most robust root move."""
        root_moves = self._root_candidate_moves(ranked_moves)
        if not root_moves:
            return None

        prior_by_move = {move: score for move, score in ranked_moves}
        root = SearchNode(board=self._copy_board(board), player_to_move=player)
        # Reverse because expansion uses pop() and we want the best move first.
        root.untried_moves = list(reversed(root_moves))

        iterations = 0
        while time.perf_counter() < deadline:
            winner = self._mcts_iteration(root, player, prior_by_move, deadline)
            if winner is None:
                break
            iterations += 1

        if not root.children:
            return root_moves[0]

        # Robust child: prioritize visit count, then win rate, then heuristic prior.
        best_child = max(
            root.children,
            key=lambda child: (
                child.visits,
                child.wins / child.visits if child.visits else -1.0,
                child.prior_score,
            ),
        )
        return best_child.move

    def _root_candidate_moves(self, ranked_moves: list[tuple[tuple[int, int], float]]) -> list[tuple[int, int]]:
        """
        Root search uses a wide but bounded candidate set.

        This is intentionally not used for tactical scans. It is only the MCTS root
        branching set after immediate wins/blocks have already been handled.
        """
        if len(ranked_moves) <= self.ROOT_CANDIDATE_LIMIT:
            return [move for move, _ in ranked_moves]

        chosen = [move for move, _ in ranked_moves[: self.ROOT_CANDIDATE_LIMIT]]

        # Add a few diversity candidates from lower ranks to reduce over-pruning risk.
        # They are deterministic positions in the ranked list rather than random picks,
        # so behavior is reproducible.
        for frac in (0.40, 0.55, 0.70, 0.85):
            idx = int(frac * (len(ranked_moves) - 1))
            move = ranked_moves[idx][0]
            if move not in chosen:
                chosen.append(move)
        return chosen

    def _mcts_iteration(
        self,
        root: SearchNode,
        root_player: int,
        prior_by_move: dict[tuple[int, int], float],
        deadline: float,
    ) -> int | None:
        """One MCTS select-expand-simulate-backpropagate iteration."""
        if time.perf_counter() >= deadline:
            return None

        node = root

        # Selection.
        while node.untried_moves == [] and node.children:
            node = self._select_child(node)
            if time.perf_counter() >= deadline:
                return None

        # Expansion.
        if node.untried_moves is None:
            node.untried_moves = list(reversed(self._tree_candidate_moves(node.board, node.player_to_move)))

        if node.untried_moves:
            move = node.untried_moves.pop()
            child_board = self._copy_board(node.board)
            self._play_in_place(child_board, move, node.player_to_move)
            child = SearchNode(
                board=child_board,
                player_to_move=self._other(node.player_to_move),
                parent=node,
                move=move,
                player_just_moved=node.player_to_move,
                prior_score=prior_by_move.get(move, 0.0),
            )
            node.children.append(child)
            node = child

        # Simulation.
        winner: int
        if node.player_just_moved is not None and self._has_player_won(node.board, node.player_just_moved):
            winner = node.player_just_moved
        else:
            winner = self._rollout(node.board, node.player_to_move, root_player, deadline)

        # Backpropagation.
        result = 1.0 if winner == root_player else 0.0
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

        return winner

    def _select_child(self, node: SearchNode) -> SearchNode:
        """UCT selection, with a small normalized prior nudge."""
        log_parent = math.log(max(1, node.visits))

        def uct(child: SearchNode) -> float:
            if child.visits == 0:
                return INF
            exploit = child.wins / child.visits
            explore = self.UCT_C * math.sqrt(log_parent / child.visits)
            # Prior only breaks close ties; MCTS statistics remain dominant.
            prior = 0.0005 * child.prior_score
            return exploit + explore + prior

        return max(node.children, key=uct)

    def _tree_candidate_moves(self, board: list[list[int]], player: int) -> list[tuple[int, int]]:
        """Lightweight move ordering for non-root MCTS expansion."""
        legal_moves = self._legal_moves(board)
        if len(legal_moves) <= self.TREE_CANDIDATE_LIMIT:
            return sorted(
                legal_moves,
                key=lambda mv: self._quick_local_score(board, mv, player),
                reverse=True,
            )

        scored = [(mv, self._quick_local_score(board, mv, player)) for mv in legal_moves]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [mv for mv, _ in scored[: self.TREE_CANDIDATE_LIMIT]]

    # -------------------------------------------------------------------------
    # Rollout policy
    # -------------------------------------------------------------------------

    def _rollout(
        self,
        start_board: list[list[int]],
        player_to_move: int,
        root_player: int,
        deadline: float,
    ) -> int:
        """
        Lightweight biased simulation.

        It avoids expensive Dijkstra/template evaluation inside playouts. Each move is
        chosen by sampling a small set of legal moves and selecting the best cheap
        local move most of the time, with some randomness preserved.
        """
        board = self._copy_board(start_board)
        player = player_to_move
        empties = self._legal_moves(board)

        # Rollout hard cap protects us if a platform machine is slow.
        while empties:
            if time.perf_counter() >= deadline:
                return self._heuristic_winner(board, root_player)

            move_index = self._choose_rollout_move_index(board, empties, player)
            move = empties[move_index]
            empties[move_index] = empties[-1]
            empties.pop()

            self._play_in_place(board, move, player)
            if self._has_player_won(board, player):
                return player
            player = self._other(player)

        return self._heuristic_winner(board, root_player)

    def _choose_rollout_move_index(
        self,
        board: list[list[int]],
        empties: list[tuple[int, int]],
        player: int,
    ) -> int:
        """Return an index into empties using a fast semi-random policy."""
        if len(empties) == 1:
            return 0

        # A small amount of pure randomness prevents deterministic rollout traps.
        if self.rng.random() < 0.18:
            return self.rng.randrange(len(empties))

        sample_size = min(self.ROLLOUT_SAMPLE_SIZE, len(empties))
        best_idx = self.rng.randrange(len(empties))
        best_score = -INF

        sampled_indices = self.rng.sample(range(len(empties)), sample_size)
        for idx in sampled_indices:
            move = empties[idx]
            if self._is_obvious_dead_rollout_cell(board, move):
                continue
            score = self._rollout_move_score(board, move, player)
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _rollout_move_score(self, board: list[list[int]], move: tuple[int, int], player: int) -> float:
        """Very cheap rollout score; no Dijkstra here."""
        opponent = self._other(player)
        score = self._quick_local_score(board, move, player)
        score += 0.70 * self._bridge_creation_count(board, move, player)
        score += 0.55 * self._opponent_bridge_attack_count(board, move, opponent)
        score += 0.25 * self._edge_orientation_bonus(board, move, player)
        return score

    def _is_obvious_dead_rollout_cell(self, board: list[list[int]], move: tuple[int, int]) -> bool:
        """
        Safe, conservative dead-cell filter for rollouts only.

        If every neighboring cell is occupied by the same player, the empty point is
        an obvious captured pocket. We skip it in rollouts, but never in tactical scans.
        """
        r, c = move
        neighbors = self._neighbors[(r, c)]
        if len(neighbors) < 3:
            return False

        first_value = None
        for nr, nc in neighbors:
            value = board[nr][nc]
            if value == EMPTY:
                return False
            if first_value is None:
                first_value = value
            elif value != first_value:
                return False
        return first_value in (PLAYER_0, PLAYER_1)

    def _heuristic_winner(self, board: list[list[int]], root_player: int) -> int:
        """Fallback winner estimate when a rollout hits the time deadline."""
        opponent = self._other(root_player)
        root_dist = self._connection_distance(board, root_player)
        opp_dist = self._connection_distance(board, opponent)
        if root_dist <= opp_dist:
            return root_player
        return opponent


if __name__ == "__main__":
    agent = HexAgent()
    agent.start()
