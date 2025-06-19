#!/usr/bin/env python3
"""
Quick test of the Take 6 system
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("Testing Take 6 Tournament System...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        from game.take6 import Take6Game
        from models.neural_network import ModelFactory
        from tournament.elo_tournament import Tournament, EloSystem
        print("   ✓ All imports successful")
        
        # Test game creation
        print("2. Testing game creation...")
        game = Take6Game(num_players=4)
        state = game.reset()
        print(f"   ✓ Game created with {state.num_players} players")
        
        # Test neural network creation
        print("3. Testing neural network creation...")
        players = ModelFactory.create_population(4, input_size=2605)
        print(f"   ✓ Created {len(players)} neural network players")
        
        # Test single game
        print("4. Testing single game...")
        valid_actions = game.get_valid_actions(0)
        card, row = players[0].get_action(state, valid_actions, game_player_id=0, training=True)
        print(f"   ✓ Player chose card {card.number}")
        
        # Test tournament
        print("5. Testing tournament...")
        elo_system = EloSystem()
        tournament = Tournament(players, elo_system)
        results = tournament.run_random_games(2, verbose=False)
        print(f"   ✓ Ran {len(results)} tournament games")
        
        # Print leaderboard
        print("6. Final leaderboard:")
        tournament.print_leaderboard(top_n=4)
        
        print("\n🎉 All tests passed! The system is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
