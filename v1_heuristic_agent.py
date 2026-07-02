import random
import math
from typing import override
from gamelib.hex.agent import Agent
from gamelib.hex.gamestate import GameState as State
from gamelib.hex.move import Move

class HeuristicHexAgent(Agent):
    """
    A simple but decent Hex agent that prefers the center of the board
    and tries to place tiles next to its own existing tiles.
    """
    
    @override
    def initialize(self, init_data: dict) -> None:
        """Initialize the agent with its player ID (0 or 1)."""
        self.player_id = init_data["player_id"]

    @override
    def get_move(self, state: State) -> Move:
        """Evaluate all empty cells and return the best one based on heuristics."""
        board_size = state.board_size
        
        empty_cells = [(r, c) for r in range(board_size) for c in range(board_size) if state.board[r][c] == -1]

        if not empty_cells:
            raise ValueError("No valid moves available.")

        best_move = None
        best_score = -float('inf')
        
        center = board_size / 2.0 

        neighbors_offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        for r, c in empty_cells:
            score = 0

            dist_to_center = math.sqrt((r - center)**2 + (c - center)**2)
            score -= dist_to_center 

            for dr, dc in neighbors_offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if state.board[nr][nc] == self.player_id:
                        score += 1.5  

            score += random.uniform(0, 0.1)

            if score > best_score:
                best_score = score
                best_move = (r, c)

        return Move(player=self.player_id, position=list(best_move))

if __name__ == "__main__":
    agent = HeuristicHexAgent()
    agent.start()