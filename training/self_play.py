"""
Training system for Take 6 neural networks using tournaments.
"""
import numpy as np
import random
from typing import List, Dict, Optional
import os
import json

from models.neural_network import Take6Player, Take6Network
from tournament.elo_tournament import Tournament
from training_logs.elo_logger import EloProgressionLogger
from training_logs.checkpoint_manager import CheckpointManager

class TournamentTrainer:
    """Trainer that uses tournament results to improve models."""
    
    def __init__(self, players: List[Take6Player]):
        self.players = players
        self.training_history = []
    
    def train_from_tournament(self, tournament_results: List[Dict]):
        """Train models based on tournament results."""
        print(f"Training models from {len(tournament_results)} games...")
        
        # Simplified training approach: adjust exploration rates based on performance
        for game_log in tournament_results:
            for i, player_idx in enumerate(game_log['players']):
                if player_idx < len(self.players):
                    player = self.players[player_idx]
                    
                    # Get player's performance in this game
                    elo_change = game_log['final_elo'][i] - game_log['initial_elo'][i]
                    
                    # Adjust exploration based on performance
                    if elo_change < 0:  # Performance declined
                        player.epsilon = min(0.3, player.epsilon + 0.01)  # Increase exploration
                    elif elo_change > 0:  # Performance improved
                        player.epsilon = max(0.05, player.epsilon - 0.005)  # Decrease exploration
        
        print("Training completed (simplified approach)")
        return {}

class AdaptiveTraining:
    """Adaptive training system that adjusts based on performance."""
    
    def __init__(self, players: List[Take6Player], tournament: Tournament, 
                 session_name: Optional[str] = None, enable_logging: bool = True,
                 enable_checkpoints: bool = True):
        self.players = players
        self.tournament = tournament
        self.trainer = TournamentTrainer(players)
        
        # Adaptive parameters
        self.performance_window = 100  # Games to consider for performance
        self.adaptation_threshold = 0.1  # Elo change threshold for adaptation
        
        # Logging and checkpoint setup
        self.enable_logging = enable_logging
        self.enable_checkpoints = enable_checkpoints
        
        if enable_logging:
            self.elo_logger = EloProgressionLogger(session_name=session_name)
            print(f"✅ ELO progression logging enabled: {self.elo_logger.csv_file}")
        
        if enable_checkpoints:
            self.checkpoint_manager = CheckpointManager(session_name=session_name)
            print(f"✅ Checkpoint system enabled: {self.checkpoint_manager.session_dir}")
        
        self.current_cycle = 0
        self.adaptation_threshold = 0.1  # Elo change threshold for adaptation
    
    def should_adapt_player(self, player: Take6Player) -> bool:
        """Determine if a player needs adaptation based on performance."""
        if player.games_played < self.performance_window:
            return False
        
        # Check if player's Elo is stagnating or declining
        return player.elo_rating < 1400  # Below average
    
    def adapt_player_strategy(self, player: Take6Player):
        """Adapt a player's strategy (e.g., increase exploration)."""
        if player.epsilon < 0.3:  # Increase exploration if too low
            player.epsilon = min(0.3, player.epsilon + 0.05)
            print(f"Increased exploration for player {player.player_id} to {player.epsilon:.3f}")
    
    def run_adaptive_cycle(self, num_matches: int = 50, target_penalty: int = 100, 
                          train_after_matches: bool = True, log_frequency: int = 10):
        """Run one cycle of adaptive training with new match structure."""
        self.current_cycle += 1
        print(f"Running adaptive training cycle {self.current_cycle} with {num_matches} matches (target: {target_penalty} penalty)...")
        
        # Run tournament matches
        results = self.tournament.run_random_matches(
            num_matches, target_penalty=target_penalty, verbose=False
        )
        
        # Log ELO progression periodically (every log_frequency matches)
        if self.enable_logging and num_matches >= log_frequency:
            # Log at intervals throughout the cycle
            for i in range(0, num_matches, log_frequency):
                self.elo_logger.log_match_results(self.players, self.current_cycle)
        
        # Train models based on results
        if train_after_matches:
            self.trainer.train_from_tournament(results)
        
        # Adapt struggling players
        for player in self.players:
            if self.should_adapt_player(player):
                self.adapt_player_strategy(player)
        
        # Log cycle summary
        if self.enable_logging:
            self.elo_logger.log_cycle_summary(self.players, self.current_cycle)
        
        return results
    
    def save_models(self, directory: str):
        """Save all player models."""
        os.makedirs(directory, exist_ok=True)
        
        # Save player statistics
        stats = {
            'player_stats': [
                {
                    'id': p.player_id,
                    'elo': p.elo_rating,
                    'games': p.games_played,
                    'total_score': p.total_score,
                    'epsilon': p.epsilon
                }
                for p in self.players
            ]
        }
        
        with open(os.path.join(directory, 'player_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved player statistics to {directory}")
    
    def create_checkpoint(self, save_top_n: int = 10, save_all: bool = False, 
                         tournament_results: Optional[Dict] = None) -> str:
        """Create a checkpoint of the current training state.
        
        Args:
            save_top_n: Number of top players to save individually
            save_all: Whether to save all players
            tournament_results: Optional tournament results to include
            
        Returns:
            Path to the created checkpoint
        """
        if not self.enable_checkpoints:
            print("⚠️  Checkpoints are disabled")
            return ""
        
        checkpoint_path = self.checkpoint_manager.create_checkpoint(
            players=self.players,
            cycle=self.current_cycle,
            save_top_n=save_top_n,
            save_all=save_all,
            tournament_results=tournament_results
        )
        
        print(f"✅ Created checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def load_best_players_from_checkpoint(self, checkpoint_name: str, top_n: int = 10) -> List[Take6Player]:
        """Load the best players from a specific checkpoint."""
        if not self.enable_checkpoints:
            raise ValueError("Checkpoints are disabled")
        
        return self.checkpoint_manager.load_best_players(checkpoint_name, top_n)
    
    def get_progression_summary(self) -> Dict:
        """Get a summary of training progress."""
        summary = {'cycle': self.current_cycle}
        
        if self.enable_logging:
            log_summary = self.elo_logger.get_progression_summary()
            summary.update(log_summary)
        
        if self.enable_checkpoints:
            checkpoints = self.checkpoint_manager.list_checkpoints()
            best_checkpoint = self.checkpoint_manager.get_best_checkpoint()
            summary.update({
                'total_checkpoints': len(checkpoints),
                'best_checkpoint': best_checkpoint
            })
        
        return summary
    
    def finalize_training(self) -> Dict:
        """Finalize the training session and create final checkpoint."""
        print("\n🏁 Finalizing training session...")
        
        summary = {}
        
        # Finalize logging
        if self.enable_logging:
            log_summary = self.elo_logger.finalize_session()
            summary['logging'] = log_summary
            print(f"✅ ELO progression logged to: {self.elo_logger.csv_file}")
        
        # Create final checkpoint
        if self.enable_checkpoints:
            final_checkpoint = self.checkpoint_manager.create_final_checkpoint(self.players)
            summary['final_checkpoint'] = final_checkpoint
            print(f"✅ Final checkpoint created: {final_checkpoint}")
        
        # Print top performers
        top_players = sorted(self.players, key=lambda p: p.elo_rating, reverse=True)[:10]
        print(f"\n🏆 Top 10 players after training:")
        for i, player in enumerate(top_players, 1):
            print(f"  {i:2d}. {player.player_id}: ELO {player.elo_rating:.1f} "
                  f"({player.games_played} games, avg score: {player.total_score/max(player.games_played,1):.1f})")
        
        return summary
    
    def load_models(self, directory: str):
        """Load player models and statistics."""
        # Load player statistics
        stats_path = os.path.join(directory, 'player_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            for i, player in enumerate(self.players):
                if i < len(stats['player_stats']):
                    player_stats = stats['player_stats'][i]
                    player.elo_rating = player_stats['elo']
                    player.games_played = player_stats['games']
                    player.total_score = player_stats['total_score']
                    player.epsilon = player_stats['epsilon']
            
            print(f"Loaded player statistics from {directory}")

# Legacy classes for compatibility
class SelfPlayTrainer:
    """Simplified trainer for compatibility."""
    
    def __init__(self, model: Take6Network, learning_rate: float = 0.001, 
                 memory_size: int = 10000, batch_size: int = 32):
        self.model = model
        # Simplified - no actual training implemented
    
    def update_target_model(self):
        """Placeholder method."""
        pass