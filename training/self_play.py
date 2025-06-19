"""
Training system for Take 6 neural networks using tournaments.
"""
import numpy as np
from typing import List, Dict
import os
import json

from models.neural_network import Take6Player, Take6Network
from tournament.elo_tournament import Tournament

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
    
    def __init__(self, players: List[Take6Player], tournament: Tournament):
        self.players = players
        self.tournament = tournament
        self.trainer = TournamentTrainer(players)
        
        # Adaptive parameters
        self.performance_window = 100  # Games to consider for performance
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
                          train_after_matches: bool = True):
        """Run one cycle of adaptive training with new match structure."""
        print(f"Running adaptive training cycle with {num_matches} matches (target: {target_penalty} penalty)...")
        
        # Run tournament matches
        results = self.tournament.run_random_matches(
            num_matches, target_penalty=target_penalty, verbose=False
        )
        
        # Train models based on results
        if train_after_matches:
            self.trainer.train_from_tournament(results)
        
        # Adapt struggling players
        for player in self.players:
            if self.should_adapt_player(player):
                self.adapt_player_strategy(player)
        
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