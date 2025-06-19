#!/usr/bin/env python3
"""
Debug script to check game state vector sizes
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.take6 import Take6Game

def check_state_vector_sizes():
    """Check the actual state vector sizes for different player counts."""
    print("Game State Vector Size Analysis")
    print("=" * 40)
    
    # Calculate theoretical size
    hand_size = 104  # Player hand (104 possible cards)
    rows_size = 4 * 6 * 104  # 4 rows * 6 positions * 104 possible cards  
    penalty_size_base = 4  # For 4 players
    round_size = 1  # Round number
    
    theoretical_4_players = hand_size + rows_size + penalty_size_base + round_size
    print(f"Theoretical size (4 players): {theoretical_4_players}")
    print(f"  Hand: {hand_size}")
    print(f"  Rows: {rows_size}")  
    print(f"  Penalties: {penalty_size_base}")
    print(f"  Round: {round_size}")
    print()
    
    # Test actual sizes
    for num_players in [2, 3, 4, 5, 6]:
        game = Take6Game(num_players=num_players)
        state = game.reset()
        
        state_vector = state.get_game_state_vector(0)
        actual_size = len(state_vector)
        
        # Calculate expected size for this player count
        penalty_size_actual = num_players
        expected_size = hand_size + rows_size + penalty_size_actual + round_size
        
        print(f"Players: {num_players}")
        print(f"  Expected: {expected_size}")
        print(f"  Actual: {actual_size}")
        print(f"  Match: {'✓' if expected_size == actual_size else '✗'}")
        print()

if __name__ == "__main__":
    check_state_vector_sizes()
