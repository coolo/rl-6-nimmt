#!/usr/bin/env python3
"""
Simple test script to validate the Take 6 tournament setup.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported."""
    try:
        print("Testing imports...")
        
        from game.take6 import Take6Game, Card, GameState
        print("✓ Game module imported successfully")
        
        from models.neural_network import Take6Player, ModelFactory, Take6Network
        print("✓ Models module imported successfully")
        
        from tournament.elo_tournament import Tournament, EloSystem
        print("✓ Tournament module imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_game():
    """Test basic game functionality."""
    try:
        print("\nTesting basic game functionality...")
        
        from game.take6 import Take6Game
        
        # Create a game
        game = Take6Game(num_players=4)
        state = game.reset()
        
        print(f"✓ Game created with {state.num_players} players")
        print(f"✓ Each player has {len(state.players_hands[0])} cards")
        print(f"✓ {len(state.rows)} rows initialized")
        
        # Test getting valid actions
        valid_actions = game.get_valid_actions(0)
        print(f"✓ Player 0 has {len(valid_actions)} valid actions")
        
        return True
    except Exception as e:
        print(f"✗ Game test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neural_network():
    """Test neural network creation."""
    try:
        print("\nTesting neural network...")
        
        from models.neural_network import ModelFactory
        
        # Create a small population
        players = ModelFactory.create_population(4, input_size=2605)
        print(f"✓ Created {len(players)} neural network players")
        
        # Test model structure
        player = players[0]
        print(f"✓ Player has Elo rating: {player.elo_rating}")
        print(f"✓ Player has epsilon: {player.epsilon}")
        
        return True
    except Exception as e:
        print(f"✗ Neural network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_game():
    """Test running a single game."""
    try:
        print("\nTesting single game execution...")
        
        from game.take6 import Take6Game
        from models.neural_network import ModelFactory
        
        # Create players
        players = ModelFactory.create_population(4, input_size=2605)
        
        # Create game
        game = Take6Game(num_players=4)
        state = game.reset()
        
        print("✓ Game and players created")
        
        # Try to get an action from the first player
        valid_actions = game.get_valid_actions(0)
        card, row = players[0].get_action(state, valid_actions, game_player_id=0, training=True)
        
        print(f"✓ Player chose card {card.number} and row {row}")
        
        return True
    except Exception as e:
        print(f"✗ Single game test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Take 6 Tournament Test Suite ===\n")
    
    tests = [
        test_imports,
        test_basic_game,
        test_neural_network,
        test_single_game
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The tournament system is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
