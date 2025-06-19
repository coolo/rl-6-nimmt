"""
Main script for Take 6 Tournament with Neural Network Players
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import random
import json
from typing import List
from datetime import datetime

# Configure TensorFlow/GPU settings early
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.take6 import Take6Game
from models.neural_network import Take6Player, ModelFactory
from tournament.elo_tournament import Tournament, EloSystem
from training.self_play import AdaptiveTraining
from analysis.checkpoint_analyzer import CheckpointAnalyzer

def setup_tensorflow():
    """Configure TensorFlow settings."""
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # GPU configuration will be handled by the models module
    # when models are first created
    print("✅ TensorFlow configuration completed")
    return True  # Return True for compatibility
    
    return len(gpus) > 0

def create_players(num_players: int = 40) -> List[Take6Player]:
    """Create initial population of players."""
    print(f"Creating {num_players} neural network players...")
    
    # Calculate input size based on game state
    # This should match the get_game_state_vector method in GameState
    input_size = 104 + (4 * 6 * 104) + 6 + 1  # Hand + Rows + Penalties(6 slots) + Round
    
    players = ModelFactory.create_population(num_players, input_size)
    
    print(f"Created {len(players)} players with input size {input_size}")
    return players

def run_training_phase(players: List[Take6Player], num_cycles: int = 10, 
                      matches_per_cycle: int = 100, target_penalty: int = 100,
                      session_name: str = None):
    """Run the main training phase with new match structure."""
    print(f"\nStarting training phase: {num_cycles} cycles, {matches_per_cycle} matches per cycle")
    print(f"Target penalty points per match: {target_penalty}")
    
    # Create tournament system
    elo_system = EloSystem(k_factor=32, initial_rating=1500.0)
    tournament = Tournament(players, elo_system)
    
    # Create adaptive training system with logging and checkpoints
    adaptive_trainer = AdaptiveTraining(
        players, tournament, 
        session_name=session_name,
        enable_logging=True,
        enable_checkpoints=True
    )
    
    all_results = []
    
    for cycle in range(num_cycles):
        print(f"\n--- Training Cycle {cycle + 1}/{num_cycles} ---")
        
        # Run matches and training
        cycle_results = adaptive_trainer.run_adaptive_cycle(
            num_matches=matches_per_cycle, 
            target_penalty=target_penalty,
            train_after_matches=True,
            log_frequency=10  # Log every 10 matches
        )
        all_results.extend(cycle_results)
        
        # Print progress
        tournament.print_leaderboard(top_n=10)
        
        # Create checkpoint every 5 cycles or on the last cycle
        if (cycle + 1) % 5 == 0 or cycle == num_cycles - 1:
            # Prepare tournament results for checkpoint
            tournament_results = {
                'cycle': cycle + 1,
                'matches_this_cycle': len(cycle_results),
                'total_matches': len(all_results),
                'leaderboard': tournament.get_leaderboard()[:20]  # Top 20 players
            }
            
            checkpoint_path = adaptive_trainer.create_checkpoint(
                save_top_n=15,  # Save top 15 players
                save_all=(cycle == num_cycles - 1),  # Save all on final cycle
                tournament_results=tournament_results
            )
            
            # Also save using the old method for compatibility
            save_dir = f"checkpoints/cycle_{cycle + 1}"
            adaptive_trainer.save_models(save_dir)
            tournament.save_results(f"{save_dir}/tournament_results.json")
            print(f"Saved checkpoint to {save_dir}")
    
    # Finalize training
    training_summary = adaptive_trainer.finalize_training()
    
    return all_results, tournament, adaptive_trainer, training_summary



def analyze_results(session_name: str = None):
    """Analyze and visualize tournament results using checkpoint data."""
    print("\nGenerating analysis and visualizations...")
    
    try:
        # Create checkpoint analyzer
        analyzer = CheckpointAnalyzer()
        
        # Load session data (will auto-detect latest if not specified)
        loaded_session = analyzer.load_session_data(session_name)
        
        # Generate comprehensive report
        analyzer.generate_comprehensive_report("analysis_output")
        
        print(f"Analysis complete for session: {loaded_session}")
        print("Check 'analysis_output/' directory for results.")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        print("This might be because no training session with logging/checkpoints was found.")
        print("Please run a training session first with the new logging system.")

def run_test_mode(checkpoint_path: str, num_seeds: int = 10):
    """Run test mode to evaluate the best player from a checkpoint against random opponents with fixed seeds."""
    print(f"\n🧪 Test Mode: Evaluating best player against random opponents")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Testing with {num_seeds} different seeds")
    print("="*60)
    
    try:
        # Import checkpoint manager
        from training_logs.checkpoint_manager import CheckpointManager
        
        # Parse checkpoint path to get session and cycle
        if '/' in checkpoint_path:
            parts = checkpoint_path.split('/')
            if 'checkpoints' in parts:
                idx = parts.index('checkpoints')
                if idx + 2 < len(parts):
                    session_name = parts[idx + 1]
                    checkpoint_name = parts[idx + 2]
                else:
                    raise ValueError("Invalid checkpoint path format")
            else:
                raise ValueError("Checkpoint path must contain 'checkpoints' directory")
        else:
            raise ValueError("Please provide full checkpoint path")
        
        # Load best player from checkpoint
        checkpoint_manager = CheckpointManager(session_name=session_name)
        best_players = checkpoint_manager.load_best_players(checkpoint_name, top_n=1)
        
        if not best_players:
            print(f"❌ No players found in checkpoint {checkpoint_path}")
            return
        
        test_player = best_players[0]
        test_player.player_id = 0  # Set to 0 for consistency
        
        print(f"✅ Loaded best player from {checkpoint_path}")
        print(f"   Player ID: {test_player.player_id}")
        
    except Exception as e:
        print(f"❌ Failed to load player: {e}")
        return
    
    # Import RandomPlayer from game module
    from game.take6 import RandomPlayer, Take6Game
    
    target_penalty = 100  # Define target penalty for games
    results_1v1 = []  # 1v1 against random
    results_1v2 = []  # 1v2 against two randoms
    
    for seed in range(num_seeds):
        print(f"\n--- Seed {seed + 1}/{num_seeds} ---")
        
        # Set random seed for reproducible tests
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Test 1: 1v1 against random player
        print("Testing 1v1 against random player...")
        players_1v1 = [test_player, RandomPlayer(1)]
        game_1v1 = Take6Game(num_players=2, target_penalty=target_penalty)
        
        # Track cumulative penalties
        cumulative_penalties_1v1 = [0, 0]
        game_count = 0
        max_games = 50
        
        # Keep playing games until someone reaches target penalty
        while max(cumulative_penalties_1v1) < target_penalty and game_count < max_games:
            state = game_1v1.reset()
            game_count += 1
            
            # Play rounds until all cards are used
            for round_num in range(10):
                if state.is_game_over():
                    break
                
                player_actions = {}
                
                # Get actions from both players
                for i, player in enumerate(players_1v1):
                    valid_actions = game_1v1.get_valid_actions(i)
                    if valid_actions:
                        if hasattr(player, 'get_action'):
                            # Neural network player
                            card, row = player.get_action(state, valid_actions, game_player_id=i, training=False)
                        else:
                            # Random player
                            card, row = player.choose_action(state, i, valid_actions)
                        player_actions[i] = (card, row)
                
                if not player_actions:
                    break
                
                # Execute round
                round_results = game_1v1.play_round(player_actions)
                
                # Update cumulative penalties
                for i, (penalty, _) in round_results.items():
                    cumulative_penalties_1v1[i] += penalty
        
        test_score_1v1 = cumulative_penalties_1v1[0]  # Test player is index 0
        random_score_1v1 = cumulative_penalties_1v1[1]
        won_1v1 = test_score_1v1 < random_score_1v1
        
        results_1v1.append({
            'seed': seed,
            'test_score': test_score_1v1,
            'opponent_score': random_score_1v1,
            'won': won_1v1,
            'rounds': game_count
        })
        
        print(f"  1v1 Result: Test={test_score_1v1}, Random={random_score_1v1}, Won={won_1v1}")
        
        # Test 2: 1v2 against two random players
        print("Testing 1v2 against two random players...")
        players_1v2 = [test_player, RandomPlayer(1), RandomPlayer(2)]
        game_1v2 = Take6Game(num_players=3, target_penalty=target_penalty)
        
        # Track cumulative penalties
        cumulative_penalties_1v2 = [0, 0, 0]
        game_count = 0
        
        # Keep playing games until someone reaches target penalty
        while max(cumulative_penalties_1v2) < target_penalty and game_count < max_games:
            state = game_1v2.reset()
            game_count += 1
            
            # Play rounds until all cards are used
            for round_num in range(10):
                if state.is_game_over():
                    break
                
                player_actions = {}
                
                # Get actions from all players
                for i, player in enumerate(players_1v2):
                    valid_actions = game_1v2.get_valid_actions(i)
                    if valid_actions:
                        if hasattr(player, 'get_action'):
                            # Neural network player
                            card, row = player.get_action(state, valid_actions, game_player_id=i, training=False)
                        else:
                            # Random player
                            card, row = player.choose_action(state, i, valid_actions)
                        player_actions[i] = (card, row)
                
                if not player_actions:
                    break
                
                # Execute round
                round_results = game_1v2.play_round(player_actions)
                
                # Update cumulative penalties
                for i, (penalty, _) in round_results.items():
                    cumulative_penalties_1v2[i] += penalty
        
        test_score_1v2 = cumulative_penalties_1v2[0]  # Test player is index 0
        opponent_scores_1v2 = cumulative_penalties_1v2[1:]
        best_opponent_score = min(opponent_scores_1v2)
        won_1v2 = test_score_1v2 < best_opponent_score
        
        results_1v2.append({
            'seed': seed,
            'test_score': test_score_1v2,
            'opponent_scores': opponent_scores_1v2,
            'best_opponent': best_opponent_score,
            'won': won_1v2,
            'rounds': game_count
        })
        
        print(f"  1v2 Result: Test={test_score_1v2}, Randoms={opponent_scores_1v2}, Won={won_1v2}")
    
    # Analyze and report results
    print(f"\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    # 1v1 Results
    wins_1v1 = sum(1 for r in results_1v1 if r['won'])
    avg_score_1v1 = sum(r['test_score'] for r in results_1v1) / len(results_1v1)
    avg_opponent_1v1 = sum(r['opponent_score'] for r in results_1v1) / len(results_1v1)
    avg_rounds_1v1 = sum(r['rounds'] for r in results_1v1) / len(results_1v1)
    
    print(f"\n📊 1v1 vs Random Player:")
    print(f"   Wins: {wins_1v1}/{num_seeds} ({wins_1v1/num_seeds*100:.1f}%)")
    print(f"   Average Score: {avg_score_1v1:.1f} vs {avg_opponent_1v1:.1f}")
    print(f"   Average Rounds: {avg_rounds_1v1:.1f}")
    
    # 1v2 Results  
    wins_1v2 = sum(1 for r in results_1v2 if r['won'])
    avg_score_1v2 = sum(r['test_score'] for r in results_1v2) / len(results_1v2)
    avg_best_opponent_1v2 = sum(r['best_opponent'] for r in results_1v2) / len(results_1v2)
    avg_rounds_1v2 = sum(r['rounds'] for r in results_1v2) / len(results_1v2)
    
    print(f"\n📊 1v2 vs Two Random Players:")
    print(f"   Wins: {wins_1v2}/{num_seeds} ({wins_1v2/num_seeds*100:.1f}%)")
    print(f"   Average Score: {avg_score_1v2:.1f} vs {avg_best_opponent_1v2:.1f} (best opponent)")
    print(f"   Average Rounds: {avg_rounds_1v2:.1f}")
    
    # Save detailed results
    test_results = {
        'test_info': {
            'checkpoint_path': checkpoint_path,
            'num_seeds': num_seeds,
            'target_penalty': target_penalty,
            'timestamp': datetime.now().isoformat()
        },
        'results_1v1': results_1v1,
        'results_1v2': results_1v2,
        'summary': {
            '1v1': {
                'wins': wins_1v1,
                'win_rate': wins_1v1/num_seeds,
                'avg_test_score': avg_score_1v1,
                'avg_opponent_score': avg_opponent_1v1,
                'avg_rounds': avg_rounds_1v1
            },
            '1v2': {
                'wins': wins_1v2,
                'win_rate': wins_1v2/num_seeds,
                'avg_test_score': avg_score_1v2,
                'avg_best_opponent_score': avg_best_opponent_1v2,
                'avg_rounds': avg_rounds_1v2
            }
        }
    }
    
    # Save results to file
    os.makedirs("test_results", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"test_results/test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    return test_results

def main():
    """Main function to run the complete tournament system."""
    parser = argparse.ArgumentParser(description="Take 6 Neural Network Tournament")
    parser.add_argument("--players", type=int, default=40, help="Number of players")
    parser.add_argument("--training-cycles", type=int, default=10, help="Number of training cycles")
    parser.add_argument("--matches-per-cycle", type=int, default=100, help="Matches per training cycle")
    parser.add_argument("--target-penalty", type=int, default=100, help="Target penalty points to end a match")
    parser.add_argument("--skip-training", action="store_true", help="Skip training phase")
    parser.add_argument("--analyze-only", action="store_true", help="Only run analysis on existing results")
    parser.add_argument("--test-mode", action="store_true", help="Run test mode to evaluate a player against random opponents")
    parser.add_argument("--test-checkpoint", type=str, help="Path to checkpoint directory for test mode (e.g., checkpoints/session/cycle_001)")
    parser.add_argument("--test-seeds", type=int, default=10, help="Number of random seeds for test mode")
    parser.add_argument("--load-checkpoint", type=str, help="Load models from checkpoint directory")
    parser.add_argument("--session-name", type=str, help="Name for this training session (also used for analysis-only mode)")
    
    args = parser.parse_args()
    
    # Setup
    setup_tensorflow()
    
    if args.analyze_only:
        analyze_results(args.session_name)
        return
    
    if args.test_mode:
        if not args.test_checkpoint:
            print("❌ Test mode requires --test-checkpoint argument")
            return
        run_test_mode(args.test_checkpoint, args.test_seeds)
        return
    
    if args.test_mode:
        run_test_mode(args.test_mode, args.num_seeds)
        return
    
    # Generate session name if not provided
    if args.session_name is None:
        args.session_name = f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🎲 Take 6 Neural Network Tournament System 🎲")
    print(f"Session: {args.session_name}")
    print("=" * 50)
    
    # Create output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create or load players
    players = create_players(args.players)
    
    if args.load_checkpoint:
        print(f"Loading models from checkpoint: {args.load_checkpoint}")
        # Note: You would implement checkpoint loading here
        # For now, we'll use fresh models
    
    training_summary = None
    
    # Training phase
    if not args.skip_training:
        training_results, tournament, adaptive_trainer, training_summary = run_training_phase(
            players, args.training_cycles, args.matches_per_cycle, args.target_penalty, args.session_name
        )
        
        # Save trained models (compatibility)
        adaptive_trainer.save_models("trained_models")
        players = adaptive_trainer.players  # Use updated players
        
        print(f"\n✅ Training phase completed!")
        if training_summary and 'logging' in training_summary:
            print(f"📊 {training_summary['logging'].get('total_matches', 'N/A')} matches logged")

    # Analysis
    analyze_results(args.session_name)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    # Print session summary
    if training_summary:
        print("📋 Session Summary:")
        print(f"   Session: {args.session_name}")
        if 'logging' in training_summary:
            log_info = training_summary['logging']
            print(f"   ELO Log: {log_info.get('files', {}).get('csv_log', 'N/A')}")
        if 'final_checkpoint' in training_summary:
            print(f"   Final Checkpoint: {training_summary['final_checkpoint']}")
        if 'best_checkpoint' in training_summary:
            best = training_summary['best_checkpoint']
            print(f"   Best Player: {best.get('player_id', 'N/A')} (ELO: {best.get('elo_rating', 'N/A')})")
    
    print("\nCheck the following directories for results:")
    print("- logs/: ELO progression and training logs")
    print("- checkpoints/: Player checkpoints and training states")
    print("- analysis_output/: Analysis and visualizations")
    print("- trained_models/: Final model weights")

if __name__ == "__main__":
    main()
