#!/usr/bin/env python3
"""
Minimal test to debug the system
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Starting minimal test...")

try:
    print("1. Testing game import...")
    from game.take6 import Take6Game
    print("   ✓ Game imported")
    
    print("2. Testing models import...")
    from models.neural_network import ModelFactory
    print("   ✓ Models imported")
    
    print("3. Creating simple test...")
    game = Take6Game(2)  # Just 2 players
    state = game.reset()
    print(f"   ✓ Game created with {len(state.players_hands)} players")
    print(f"   ✓ Player 0 has {len(state.players_hands[0])} cards")
    
    print("4. Creating players...")
    players = ModelFactory.create_population(2, input_size=2608)
    print(f"   ✓ Created {len(players)} players")
    
    print("5. Testing game state...")
    state_vector = state.get_game_state_vector(0)
    print(f"   ✓ State vector size: {len(state_vector)}")
    
    print("✅ Basic test completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
