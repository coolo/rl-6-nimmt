"""
Main script for Take 6 Tournament with Neural Network Players
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from typing import List

# Configure TensorFlow/GPU settings early
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.take6 import Take6Game
from models.neural_network import Take6Player, ModelFactory
from tournament.elo_tournament import Tournament, EvolutionaryTournament, EloSystem
from training.self_play import AdaptiveTraining
from analysis.visualize_results import TournamentAnalyzer

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

def run_evolutionary_phase(players: List[Take6Player], num_generations: int = 5,
                          matches_per_generation: int = 250, target_penalty: int = 100):
    """Run evolutionary tournament phase with new match structure."""
    print(f"\nStarting evolutionary phase: {num_generations} generations")
    print(f"Target penalty points per match: {target_penalty}")
    
    # Create evolutionary tournament
    elo_system = EloSystem(k_factor=40, initial_rating=1500.0)
    evo_tournament = EvolutionaryTournament(
        players, elo_system, 
        selection_ratio=0.3, 
        mutation_rate=0.15
    )
    
    for generation in range(num_generations):
        print(f"\n--- Generation {generation + 1}/{num_generations} ---")
        
        # Run tournament matches
        results = evo_tournament.run_random_matches(
            matches_per_generation, target_penalty=target_penalty, verbose=False
        )
        
        # Print current leaderboard
        evo_tournament.print_leaderboard(top_n=15)
        
        # Evolve population (except for last generation)
        if generation < num_generations - 1:
            evo_tournament.evolve_population()
        
        # Save generation results
        save_dir = f"evolution/generation_{generation + 1}"
        os.makedirs(save_dir, exist_ok=True)
        evo_tournament.save_results(f"{save_dir}/results.json")
    
    return evo_tournament

def run_final_tournament(players: List[Take6Player], num_matches: int = 500, 
                        target_penalty: int = 100):
    """Run final comprehensive tournament with new match structure."""
    print(f"\nRunning final tournament with {num_matches} matches (target: {target_penalty} penalty)...")
    
    elo_system = EloSystem(k_factor=16, initial_rating=1500.0)  # Lower K for stability
    final_tournament = Tournament(players, elo_system)
    
    # Run comprehensive tournament
    results = final_tournament.run_random_matches(
        num_matches, target_penalty=target_penalty, verbose=True
    )
    
    # Print final standings
    print("\n" + "="*60)
    print("FINAL TOURNAMENT RESULTS")
    print("="*60)
    final_tournament.print_leaderboard(top_n=20)
    
    # Save final results
    os.makedirs("final_results", exist_ok=True)
    final_tournament.save_results("final_results/tournament_results.json")
    
    return final_tournament, results

def analyze_results():
    """Analyze and visualize tournament results."""
    print("\nGenerating analysis and visualizations...")
    
    # Find the most recent results file
    results_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file == "tournament_results.json":
                results_files.append(os.path.join(root, file))
    
    if not results_files:
        print("No tournament results found for analysis")
        return
    
    # Use the most recent results
    latest_results = max(results_files, key=os.path.getmtime)
    print(f"Analyzing results from: {latest_results}")
    
    analyzer = TournamentAnalyzer(latest_results)
    analyzer.generate_report("analysis_output")
    
    print("Analysis complete! Check 'analysis_output/' directory for results.")

def main():
    """Main function to run the complete tournament system."""
    parser = argparse.ArgumentParser(description="Take 6 Neural Network Tournament")
    parser.add_argument("--players", type=int, default=40, help="Number of players")
    parser.add_argument("--training-cycles", type=int, default=10, help="Number of training cycles")
    parser.add_argument("--matches-per-cycle", type=int, default=100, help="Matches per training cycle")
    parser.add_argument("--evolutionary-generations", type=int, default=5, help="Number of evolutionary generations")
    parser.add_argument("--matches-per-generation", type=int, default=250, help="Matches per evolutionary generation")
    parser.add_argument("--final-matches", type=int, default=500, help="Number of final tournament matches")
    parser.add_argument("--target-penalty", type=int, default=100, help="Target penalty points to end a match")
    parser.add_argument("--skip-training", action="store_true", help="Skip training phase")
    parser.add_argument("--skip-evolution", action="store_true", help="Skip evolutionary phase")
    parser.add_argument("--analyze-only", action="store_true", help="Only run analysis on existing results")
    parser.add_argument("--load-checkpoint", type=str, help="Load models from checkpoint directory")
    parser.add_argument("--session-name", type=str, help="Name for this training session")
    
    args = parser.parse_args()
    
    # Setup
    setup_tensorflow()
    
    if args.analyze_only:
        analyze_results()
        return
    
    # Generate session name if not provided
    if args.session_name is None:
        from datetime import datetime
        args.session_name = f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🎲 Take 6 Neural Network Tournament System 🎲")
    print(f"Session: {args.session_name}")
    print("=" * 50)
    
    # Create output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("evolution", exist_ok=True)
    os.makedirs("final_results", exist_ok=True)
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
    
    # Evolutionary phase
    if not args.skip_evolution:
        evo_tournament = run_evolutionary_phase(
            players, args.evolutionary_generations, args.matches_per_generation, args.target_penalty
        )
        players = evo_tournament.players  # Use evolved players
    
    # Final tournament
    final_tournament, final_results = run_final_tournament(players, args.final_matches, args.target_penalty)
    
    # Analysis
    analyze_results()
    
    print("\n" + "="*60)
    print("TOURNAMENT COMPLETE!")
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
    print("- final_results/: Final tournament results")
    print("- analysis_output/: Analysis and visualizations")
    print("- trained_models/: Final model weights")
    print("- evolution/: Evolutionary phase results")

if __name__ == "__main__":
    main()
