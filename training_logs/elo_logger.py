"""
ELO progression logging system for tracking player performance during training.
"""
import os
import json
import csv
import logging
from datetime import datetime
from typing import List, Dict, Optional, TYPE_CHECKING
from collections import defaultdict

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from models.neural_network import Take6Player

class EloProgressionLogger:
    """Logger to track ELO progression of players during training."""
    
    def __init__(self, log_dir: str = "logs", session_name: Optional[str] = None):
        """Initialize the ELO progression logger.
        
        Args:
            log_dir: Directory to store log files
            session_name: Name for this training session (auto-generated if None)
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate session name if not provided
        if session_name is None:
            session_name = f"training_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_name = session_name
        
        # File paths
        self.csv_file = os.path.join(log_dir, f"{session_name}_elo_progression.csv")
        self.json_file = os.path.join(log_dir, f"{session_name}_detailed_log.json")
        
        # Initialize data structures
        self.progression_data = []
        self.match_counter = 0
        self.cycle_counter = 0
        
        # Initialize CSV file with headers
        self._initialize_csv()
        
        # Initialize detailed log
        self.detailed_log = {
            'session_info': {
                'session_name': session_name,
                'start_time': datetime.now().isoformat(),
                'log_directory': log_dir
            },
            'progression_log': []
        }
        
        # Setup logging
        self.logger = logging.getLogger('EloProgressionLogger')
        handler = logging.FileHandler(os.path.join(log_dir, f"{session_name}_training.log"))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Started ELO progression logging for session: {session_name}")
    
    def _initialize_csv(self):
        """Initialize CSV file with headers."""
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'match_number', 'cycle', 'player_id', 'elo_rating', 
                'elo_change', 'games_played', 'avg_score', 'total_score',
                'epsilon', 'timestamp'
            ])
    
    def log_match_results(self, players: List["Take6Player"], cycle: int = None):
        """Log ELO ratings after a match.
        
        Args:
            players: List of players to log
            cycle: Current training cycle (optional)
        """
        self.match_counter += 1
        if cycle is not None:
            self.cycle_counter = cycle
        
        timestamp = datetime.now().isoformat()
        
        # Prepare data for this match
        match_data = {
            'match_number': self.match_counter,
            'cycle': self.cycle_counter,
            'timestamp': timestamp,
            'players': []
        }
        
        # Log each player's data
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            for player in players:
                # Calculate ELO change (if we have previous data)
                elo_change = self._calculate_elo_change(player)
                
                # Write to CSV
                writer.writerow([
                    self.match_counter, self.cycle_counter, player.player_id,
                    round(player.elo_rating, 2), round(elo_change, 2),
                    player.games_played, 
                    round(player.total_score / max(player.games_played, 1), 2),
                    player.total_score, 
                    round(player.epsilon, 3), timestamp
                ])
                
                # Add to detailed log
                player_data = {
                    'player_id': player.player_id,
                    'elo_rating': round(player.elo_rating, 2),
                    'elo_change': round(elo_change, 2),
                    'games_played': player.games_played,
                    'avg_score': round(player.total_score / max(player.games_played, 1), 2),
                    'total_score': player.total_score,
                    'epsilon': round(player.epsilon, 3)
                }
                match_data['players'].append(player_data)
        
        # Add to detailed log
        self.detailed_log['progression_log'].append(match_data)
        
        # Save detailed log periodically
        if self.match_counter % 10 == 0:
            self._save_detailed_log()
        
        # Log significant changes
        top_performers = sorted(players, key=lambda p: p.elo_rating, reverse=True)[:3]
        self.logger.info(f"Match {self.match_counter} completed. Top performers: " + 
                        ", ".join([f"{p.player_id}({p.elo_rating:.1f})" for p in top_performers]))
    
    def _calculate_elo_change(self, player: "Take6Player") -> float:
        """Calculate ELO change since last logged match for this player."""
        # Look for the last entry for this player
        for entry in reversed(self.progression_data):
            if entry.get('player_id') == player.player_id:
                return player.elo_rating - entry.get('elo_rating', player.elo_rating)
        return 0.0  # No previous data
    
    def log_cycle_summary(self, players: List["Take6Player"], cycle: int):
        """Log summary statistics for a completed cycle.
        
        Args:
            players: All players in the tournament
            cycle: Cycle number that just completed
        """
        # Calculate cycle statistics
        elo_ratings = [p.elo_rating for p in players]
        avg_elo = sum(elo_ratings) / len(elo_ratings)
        max_elo = max(elo_ratings)
        min_elo = min(elo_ratings)
        
        # Find best and worst performers
        best_player = max(players, key=lambda p: p.elo_rating)
        worst_player = min(players, key=lambda p: p.elo_rating)
        
        summary = {
            'cycle': cycle,
            'timestamp': datetime.now().isoformat(),
            'total_matches': self.match_counter,
            'statistics': {
                'avg_elo': round(avg_elo, 2),
                'max_elo': round(max_elo, 2),
                'min_elo': round(min_elo, 2),
                'elo_spread': round(max_elo - min_elo, 2),
                'best_player': {
                    'id': best_player.player_id,
                    'elo': round(best_player.elo_rating, 2),
                    'games': best_player.games_played
                },
                'worst_player': {
                    'id': worst_player.player_id,
                    'elo': round(worst_player.elo_rating, 2),
                    'games': worst_player.games_played
                }
            }
        }
        
        # Log cycle summary
        self.logger.info(f"Cycle {cycle} completed: avg_elo={avg_elo:.1f}, "
                        f"spread={max_elo-min_elo:.1f}, best={best_player.player_id}({max_elo:.1f})")
        
        # Add to detailed log
        if 'cycle_summaries' not in self.detailed_log:
            self.detailed_log['cycle_summaries'] = []
        self.detailed_log['cycle_summaries'].append(summary)
        
        self._save_detailed_log()
    
    def _save_detailed_log(self):
        """Save the detailed log to JSON file."""
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(self.detailed_log, f, indent=2)
    
    def get_progression_summary(self) -> Dict:
        """Get a summary of the progression data."""
        if not os.path.exists(self.csv_file):
            return {'error': 'No progression data found'}
        
        # Read CSV data
        import pandas as pd
        try:
            df = pd.read_csv(self.csv_file)
            
            summary = {
                'total_matches': self.match_counter,
                'total_cycles': self.cycle_counter,
                'unique_players': df['player_id'].nunique(),
                'elo_statistics': {
                    'current_avg': df.groupby('player_id')['elo_rating'].last().mean(),
                    'current_max': df.groupby('player_id')['elo_rating'].last().max(),
                    'current_min': df.groupby('player_id')['elo_rating'].last().min(),
                    'highest_ever': df['elo_rating'].max(),
                    'lowest_ever': df['elo_rating'].min()
                },
                'files': {
                    'csv_log': self.csv_file,
                    'json_log': self.json_file,
                    'training_log': os.path.join(self.log_dir, f"{self.session_name}_training.log")
                }
            }
            
            return summary
        except ImportError:
            return {'error': 'pandas not available for summary generation'}
        except Exception as e:
            return {'error': f'Error generating summary: {str(e)}'}
    
    def finalize_session(self):
        """Finalize the logging session."""
        self.detailed_log['session_info']['end_time'] = datetime.now().isoformat()
        self.detailed_log['session_info']['total_matches'] = self.match_counter
        self.detailed_log['session_info']['total_cycles'] = self.cycle_counter
        self._save_detailed_log()
        
        self.logger.info(f"Session {self.session_name} finalized. Total matches: {self.match_counter}")
        
        return self.get_progression_summary()
