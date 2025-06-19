"""
Checkpoint system for saving and loading the best performing players during training.
"""
import os
import json
import shutil
import pickle
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from pathlib import Path

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from models.neural_network import Take6Player

class CheckpointManager:
    """Manager for creating and loading player checkpoints during training."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", session_name: Optional[str] = None):
        """Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Base directory for checkpoints
            session_name: Name for this training session
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if session_name is None:
            session_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_name = session_name
        
        # Create session directory
        self.session_dir = os.path.join(checkpoint_dir, session_name)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('CheckpointManager')
        handler = logging.FileHandler(os.path.join(self.session_dir, "checkpoint.log"))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Track checkpoints
        self.checkpoint_history = []
        self.best_player_history = []
        
        self.logger.info(f"Initialized checkpoint manager for session: {session_name}")
    
    def create_checkpoint(self, players: List["Take6Player"], cycle: int, 
                         save_top_n: int = 10, save_all: bool = False,
                         tournament_results: Optional[Dict] = None) -> str:
        """Create a checkpoint of the current training state.
        
        Args:
            players: List of all players
            cycle: Current training cycle
            save_top_n: Number of top players to save individually
            save_all: Whether to save all players (in addition to top performers)
            tournament_results: Optional tournament results to include
            
        Returns:
            Path to the created checkpoint directory
        """
        # Create checkpoint directory
        checkpoint_name = f"cycle_{cycle:03d}"
        checkpoint_path = os.path.join(self.session_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        
        # Sort players by ELO rating
        sorted_players = sorted(players, key=lambda p: p.elo_rating, reverse=True)
        
        # Create checkpoint metadata
        checkpoint_info = {
            'checkpoint_name': checkpoint_name,
            'cycle': cycle,
            'timestamp': timestamp,
            'total_players': len(players),
            'top_n_saved': save_top_n,
            'all_players_saved': save_all,
            'statistics': self._calculate_population_stats(sorted_players),
            'top_players': []
        }
        
        # Save top N players individually
        top_players_dir = os.path.join(checkpoint_path, "top_players")
        os.makedirs(top_players_dir, exist_ok=True)
        
        for i, player in enumerate(sorted_players[:save_top_n]):
            player_file = os.path.join(top_players_dir, f"rank_{i+1:02d}_{player.player_id}.pkl")
            self._save_player(player, player_file)
            
            # Add to checkpoint info
            checkpoint_info['top_players'].append({
                'rank': i + 1,
                'player_id': player.player_id,
                'elo_rating': round(player.elo_rating, 2),
                'games_played': player.games_played,
                'avg_score': round(player.total_score / max(player.games_played, 1), 2),
                'file': os.path.basename(player_file)
            })
        
        # Save all players if requested
        if save_all:
            all_players_file = os.path.join(checkpoint_path, "all_players.pkl")
            with open(all_players_file, 'wb') as f:
                pickle.dump(sorted_players, f)
            checkpoint_info['all_players_file'] = "all_players.pkl"
        
        # Save player statistics in JSON format
        stats_file = os.path.join(checkpoint_path, "player_stats.json")
        player_stats = {
            'timestamp': timestamp,
            'cycle': cycle,
            'players': [
                {
                    'player_id': p.player_id,
                    'elo_rating': round(p.elo_rating, 2),
                    'games_played': p.games_played,
                    'total_score': p.total_score,
                    'avg_score': round(p.total_score / max(p.games_played, 1), 2),
                    'epsilon': round(p.epsilon, 3)
                }
                for p in sorted_players
            ]
        }
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(player_stats, f, indent=2)
        
        # Save tournament results if provided
        if tournament_results:
            results_file = os.path.join(checkpoint_path, "tournament_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(tournament_results, f, indent=2)
            checkpoint_info['tournament_results_file'] = "tournament_results.json"
        
        # Save checkpoint info
        info_file = os.path.join(checkpoint_path, "checkpoint_info.json")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        # Update checkpoint history
        self.checkpoint_history.append(checkpoint_info)
        
        # Track best player progression
        best_player = sorted_players[0]
        self.best_player_history.append({
            'cycle': cycle,
            'timestamp': timestamp,
            'player_id': best_player.player_id,
            'elo_rating': round(best_player.elo_rating, 2),
            'games_played': best_player.games_played,
            'checkpoint_path': checkpoint_path
        })
        
        # Save session history
        self._save_session_history()
        
        self.logger.info(f"Created checkpoint {checkpoint_name}: "
                        f"top player {best_player.player_id} (ELO: {best_player.elo_rating:.1f})")
        
        return checkpoint_path
    
    def _save_player(self, player: "Take6Player", file_path: str):
        """Save a single player to a file."""
        player_data = {
            'player_id': player.player_id,
            'elo_rating': player.elo_rating,
            'games_played': player.games_played,
            'total_score': player.total_score,
            'epsilon': player.epsilon,
            'model_weights': player.model.get_weights() if hasattr(player.model, 'get_weights') else None,
            'model_config': player.model.get_config() if hasattr(player.model, 'get_config') else None
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(player_data, f)
    
    def load_player(self, file_path: str) -> "Take6Player":
        """Load a player from a checkpoint file."""
        with open(file_path, 'rb') as f:
            player_data = pickle.load(f)
        
        # Import at runtime to avoid circular import
        from models.neural_network import Take6Player, Take6Network
        
        # Reconstruct player (this is simplified - you may need to adapt based on your model structure)
        
        # Create new player
        input_size = 104 + (4 * 6 * 104) + 6 + 1  # Should match your model architecture
        model = Take6Network(input_size=input_size)
        player = Take6Player(player_data['player_id'], model)
        
        # Restore state
        player.elo_rating = player_data['elo_rating']
        player.games_played = player_data['games_played']
        player.total_score = player_data['total_score']
        player.epsilon = player_data['epsilon']
        
        # Restore model weights if available
        if player_data.get('model_weights'):
            try:
                player.model.set_weights(player_data['model_weights'])
            except Exception as e:
                self.logger.warning(f"Could not restore model weights for {player.player_id}: {e}")
        
        return player
    
    def load_best_players(self, checkpoint_name: str, top_n: int = 10) -> List["Take6Player"]:
        """Load the best players from a specific checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint (e.g., "cycle_005")
            top_n: Number of top players to load
            
        Returns:
            List of loaded players, sorted by ELO rating
        """
        checkpoint_path = os.path.join(self.session_dir, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found")
        
        top_players_dir = os.path.join(checkpoint_path, "top_players")
        if not os.path.exists(top_players_dir):
            raise FileNotFoundError(f"Top players directory not found in {checkpoint_name}")
        
        # Find player files
        player_files = []
        for file in os.listdir(top_players_dir):
            if file.endswith('.pkl'):
                player_files.append(os.path.join(top_players_dir, file))
        
        # Sort by rank (embedded in filename)
        player_files.sort()
        
        # Load players
        loaded_players = []
        for i, file_path in enumerate(player_files[:top_n]):
            try:
                player = self.load_player(file_path)
                loaded_players.append(player)
            except Exception as e:
                self.logger.error(f"Failed to load player from {file_path}: {e}")
        
        self.logger.info(f"Loaded {len(loaded_players)} players from checkpoint {checkpoint_name}")
        return loaded_players
    
    def get_best_checkpoint(self) -> Optional[Dict]:
        """Get information about the checkpoint with the highest ELO best player."""
        if not self.best_player_history:
            return None
        
        best_checkpoint = max(self.best_player_history, key=lambda x: x['elo_rating'])
        return best_checkpoint
    
    def _calculate_population_stats(self, sorted_players: List["Take6Player"]) -> Dict:
        """Calculate statistics for the current population."""
        elo_ratings = [p.elo_rating for p in sorted_players]
        
        return {
            'avg_elo': round(sum(elo_ratings) / len(elo_ratings), 2),
            'max_elo': round(max(elo_ratings), 2),
            'min_elo': round(min(elo_ratings), 2),
            'elo_spread': round(max(elo_ratings) - min(elo_ratings), 2),
            'median_elo': round(elo_ratings[len(elo_ratings)//2], 2),
            'std_elo': round(np.std(elo_ratings), 2) if len(elo_ratings) > 1 else 0.0
        }
    
    def _save_session_history(self):
        """Save the session history to file."""
        history_file = os.path.join(self.session_dir, "session_history.json")
        session_data = {
            'session_name': self.session_name,
            'checkpoint_history': self.checkpoint_history,
            'best_player_history': self.best_player_history,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)
    
    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints for this session."""
        checkpoints = []
        for item in os.listdir(self.session_dir):
            checkpoint_path = os.path.join(self.session_dir, item)
            if os.path.isdir(checkpoint_path) and item.startswith('cycle_'):
                info_file = os.path.join(checkpoint_path, "checkpoint_info.json")
                if os.path.exists(info_file):
                    with open(info_file, 'r', encoding='utf-8') as f:
                        checkpoints.append(json.load(f))
        
        return sorted(checkpoints, key=lambda x: x['cycle'])
    
    def create_final_checkpoint(self, players: List["Take6Player"], 
                               tournament_results: Optional[Dict] = None) -> str:
        """Create a final checkpoint with all the best players."""
        return self.create_checkpoint(
            players=players, 
            cycle=999,  # Special cycle number for final checkpoint
            save_top_n=20,  # Save more top players for final
            save_all=True,  # Save all players
            tournament_results=tournament_results
        )


