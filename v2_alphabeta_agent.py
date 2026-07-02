import time
import random
import heapq
from typing import override

from gamelib.hex.agent import Agent
from gamelib.hex.gamestate import GameState as State
from gamelib.hex.move import Move

class TournamentHexAgent(Agent):
    """
    An advanced Hex agent utilizing Iterative Deepening Alpha-Beta search 
    combined with a Path-Extraction Heuristic to bypass the Python 5-second limit.
    """

    @override
    def initialize(self, init_data: dict) -> None:
        self.player_id = init_data["player_id"]
        self.opp_id = 1 - self.player_id
        # Safe time limit to ensure we return before the 5.0s platform kill switch
        self.time_limit = 4.8 
        self.max_branching = 12

    @override
    def get_move(self, state: State) -> Move:
        self.start_time = time.time()
        size = state.board_size
        
        # CRITICAL FIX: Create a local copy of the board!
        # The game engine passes `state.board` by reference. If we mutate it during 
        # our search and throw a TimeoutError, we permanently corrupt the engine.
        board = [row[:] for row in state.board]
        
        # Identify all empty cells
        empty_cells = [(r, c) for r in range(size) for c in range(size) if board[r][c] == -1]
        
        # 1. Opening Book (No-Swap Optimization)
        if len(empty_cells) == size * size:
            return Move(player=self.player_id, position=[size // 2, size // 2])
        if len(empty_cells) == size * size - 1:
            if board[size // 2][size // 2] != -1:
                # If opponent took center, take a strong defensive adjacency
                return Move(player=self.player_id, position=[size // 2, (size // 2) + 1])
            else:
                return Move(player=self.player_id, position=[size // 2, size // 2])

        # 2. Candidate Generation & Shallow Evaluation
        opp_c_init, _ = self.get_distance_and_path(board, self.opp_id, size)
        
        candidates_scores = []
        for r, c in empty_cells:
            if time.time() - self.start_time > self.time_limit:
                break
                
            # Simulate our move on the LOCAL board
            board[r][c] = self.player_id
            my_c, _ = self.get_distance_and_path(board, self.player_id, size)
            
            # Immediate win detection
            if my_c == 0:
                return Move(player=self.player_id, position=[r, c])
                
            opp_c, _ = self.get_distance_and_path(board, self.opp_id, size)
            board[r][c] = -1 # Revert simulation
            
            # Tactical check: If opponent is 1 move away, and our move doesn't block them, discard
            if opp_c_init == 1 and opp_c <= 1:
                score = -5000 
            else:
                score = opp_c - my_c
                score -= (abs(r - size/2) + abs(c - size/2)) * 0.05
                
            candidates_scores.append((score, (r, c)))

        # Sort moves by highest tactical score
        candidates_scores.sort(key=lambda x: x[0], reverse=True)
        top_moves = [m[1] for m in candidates_scores[:self.max_branching]]
        
        best_move = top_moves[0] if top_moves else random.choice(empty_cells)
        
        # 3. Iterative Deepening Alpha-Beta Search
        depth = 2
        try:
            while depth <= min(10, len(empty_cells)):
                move, score = self.ab_search(board, size, depth, -float('inf'), float('inf'), True, top_moves)
                if move is not None:
                    best_move = move
                # Stop deepening if we found a forced win/loss
                if score > 9000 or score < -9000:
                    break
                depth += 1
        except TimeoutError:
            # Time limit reached safely! Our local board copy protects the engine.
            pass
            
        return Move(player=self.player_id, position=list(best_move))

    def ab_search(self, board, size, depth, alpha, beta, maximizing, allowed_moves):
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError()
            
        my_cost, my_path = self.get_distance_and_path(board, self.player_id, size)
        opp_cost, opp_path = self.get_distance_and_path(board, self.opp_id, size)
        
        if my_cost == 0: return None, 10000 + depth
        if opp_cost == 0: return None, -10000 - depth
        if depth == 0: return None, opp_cost - my_cost
        
        if allowed_moves is None:
            candidates = list(set(my_path + opp_path))
            if not candidates:
                return None, opp_cost - my_cost
        else:
            candidates = allowed_moves
            
        if maximizing:
            max_eval = -float('inf')
            best_m = candidates[0]
            for r, c in candidates:
                if board[r][c] != -1: continue 
                
                board[r][c] = self.player_id
                try:
                    _, eval_score = self.ab_search(board, size, depth - 1, alpha, beta, False, None)
                finally:
                    board[r][c] = -1 # Safe unwinding guarantee
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_m = (r, c)
                alpha = max(alpha, eval_score)
                if beta <= alpha: break
            return best_m, max_eval
        else:
            min_eval = float('inf')
            best_m = candidates[0]
            for r, c in candidates:
                if board[r][c] != -1: continue
                
                board[r][c] = self.opp_id
                try:
                    _, eval_score = self.ab_search(board, size, depth - 1, alpha, beta, True, None)
                finally:
                    board[r][c] = -1
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_m = (r, c)
                beta = min(beta, eval_score)
                if beta <= alpha: break
            return best_m, min_eval

    def get_distance_and_path(self, board, player_id, size):
        pq = []
        min_cost = [[float('inf')] * size for _ in range(size)]
        parent = [[None] * size for _ in range(size)]
        
        # FIX: Correctly map player orientation based on standard platform rules
        # Player 0 = Left-Right | Player 1 = Top-Bottom
        is_top_bottom = (player_id == 1)
        
        for i in range(size):
            r, c = (0, i) if is_top_bottom else (i, 0)
            cell = board[r][c]
            if cell == player_id:
                heapq.heappush(pq, (0, r, c))
                min_cost[r][c] = 0
            elif cell == -1:
                heapq.heappush(pq, (1, r, c))
                min_cost[r][c] = 1

        best_cost = float('inf')
        best_end = None

        while pq:
            cost, r, c = heapq.heappop(pq)
            
            if cost > min_cost[r][c]:
                continue
                
            # Target check based on orientation
            target_reached = (r == size - 1) if is_top_bottom else (c == size - 1)
            if target_reached:
                best_cost = cost
                best_end = (r, c)
                break

            for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    cell = board[nr][nc]
                    if cell == 1 - player_id:
                        continue 
                        
                    new_cost = cost + (0 if cell == player_id else 1)
                    if new_cost < min_cost[nr][nc]:
                        min_cost[nr][nc] = new_cost
                        parent[nr][nc] = (r, c)
                        heapq.heappush(pq, (new_cost, nr, nc))
                        
        if best_cost == float('inf'):
            return float('inf'), []
            
        path = []
        curr = best_end
        while curr is not None:
            r, c = curr
            if board[r][c] == -1:
                path.append((r, c))
            curr = parent[r][c]
            
        return best_cost, path

if __name__ == "__main__":
    agent = TournamentHexAgent()
    agent.start()