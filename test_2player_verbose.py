#!/usr/bin/env python3
"""
Test script to run a verbose match between 2 players using tournament logic
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.neural_network import ModelFactory
from tournament.elo_tournament import Tournament, EloSystem

def run_verbose_2player_match():
    """Run a single match between 2 players using the actual tournament system."""
    print("=== Take 6 Tournament: 2-Player Verbose Match ===\n")
    
    # Create 2 diverse players
    print("Creating players...")
    players = ModelFactory.create_population(2, input_size=2608)
    
    print(f"Player 0: Elo {players[0].elo_rating:.1f}, Epsilon {players[0].epsilon:.3f}")
    print(f"Player 1: Elo {players[1].elo_rating:.1f}, Epsilon {players[1].epsilon:.3f}")
    
    # Create tournament system
    elo_system = EloSystem(k_factor=32, initial_rating=1500.0)
    tournament = Tournament(players, elo_system)
    
    # Record initial Elo ratings
    initial_elos = [p.elo_rating for p in players]
    
    # Use the actual tournament method to run a single match
    target_penalty = 100
    print(f"\nStarting match (target: {target_penalty} penalty points)")
    print("=" * 60)
    
    # Run the match using the exact same logic as the main tournament
    match_result = tournament.run_single_match(
        players, 
        min_players=2, 
        max_players=2, 
        target_penalty=target_penalty, 
        verbose=True
    )
    
    # Print final results using the match result data
    print("\n" + "=" * 60)
    print("🏆 MATCH RESULTS")
    print("=" * 60)
    
    print(f"Total games played: {match_result['total_games']}")
    print(f"Target penalty points: {target_penalty}")
    print()
    
    for i, player in enumerate(players):
        elo_change = player.elo_rating - initial_elos[i]
        status = "🥇 WINNER" if i == match_result['winner'] else "🥈"
        print(f"Player {i} {status}")
        print(f"  Final penalty points: {match_result['final_scores'][i]}")
        print(f"  Elo rating: {initial_elos[i]:.1f} -> {player.elo_rating:.1f} ({elo_change:+.1f})")
        print(f"  Games played: {player.games_played}")
        print()
    
    # Show detailed game breakdown
    print("📋 DETAILED BREAKDOWN:")
    print(f"Match involved {len(match_result['games'])} individual games")
    for i, game in enumerate(match_result['games']):
        print(f"  Game {i+1}: {len(game['rounds'])} rounds, penalties {game['penalty_totals_start']} -> {game['penalty_totals_end']}")
    
    return match_result

def main():
    """Run the verbose 2-player match test."""
    try:
        print("Choose test mode:")
        print("1. Verbose 2-player match")
        print("2. Simple tournament test (4 players, 3 matches)")
        
        choice = input("Enter choice (1 or 2, or just press Enter for option 1): ").strip()
        
        if choice == "2":
            results = run_simple_tournament_test()
        else:
            results = run_verbose_2player_match()
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_simple_tournament_test():
    """Run a simple test using the tournament's existing random match functionality."""
    print("=== Simple Tournament Test (Using Actual Tournament Logic) ===\n")
    
    # Create players
    players = ModelFactory.create_population(4, input_size=2608)
    print(f"Created {len(players)} players")
    
    # Create tournament
    elo_system = EloSystem(k_factor=32, initial_rating=1500.0)
    tournament = Tournament(players, elo_system)
    
    # Run a few random matches using the exact same logic as main tournament
    print("Running 3 random matches...")
    results = tournament.run_random_matches(
        num_matches=3, 
        target_penalty=100, 
        min_players=2, 
        max_players=4, 
        verbose=True
    )
    
    print(f"\n✅ Completed {len(results)} matches successfully!")
    tournament.print_leaderboard(top_n=4)
    
    return results

main()
