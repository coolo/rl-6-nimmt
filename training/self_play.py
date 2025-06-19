"""
Training system for Take 6 neural networks using self-play and tournaments.
"""
import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple
import random
from collections import deque
import pickle
import os

from game.take6 import Take6Game, GameState
from models.neural_network import Take6Player, Take6Network
from tournament.elo_tournament import Tournament

class SelfPlayTrainer:
    """Trainer using self-play and experience replay."""
    
    def __init__(self, model: Take6Network, learning_rate: float = 0.001, 
                 memory_size: int = 10000, batch_size: int = 32):
        self.model = model
        self.target_model = Take6Network(model.input_size, model.hidden_size, 
                                       name=f"{model.name}_target")
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        # Optimizers for different heads
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Loss functions
        self.card_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.row_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.value_loss_fn = tf.keras.losses.MeanSquaredError()
    
    def store_experience(self, state: np.ndarray, action: Dict, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample_batch(self) -> List[Tuple]:
        """Sample a batch of experiences from memory."""
        return random.sample(self.memory, min(len(self.memory), self.batch_size))
    
    @tf.function
    def train_step(self, states, card_targets, row_targets, value_targets):
        """Perform one training step."""
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            
            # Calculate losses
            card_loss = self.card_loss_fn(card_targets, predictions['card_probs'])
            row_loss = self.row_loss_fn(row_targets, predictions['row_probs'])
            value_loss = self.value_loss_fn(value_targets, predictions['value'])
            
            total_loss = card_loss + row_loss + 0.5 * value_loss
        
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss, card_loss, row_loss, value_loss
    
    def train_from_memory(self) -> Dict[str, float]:
        """Train the model using experiences from memory."""
        if len(self.memory) < self.batch_size:
            return {}
        
        batch = self.sample_batch()
        
        states = []
        card_targets = []
        row_targets = []
        value_targets = []
        
        for state, action, reward, next_state, done in batch:
            states.append(state)
            
            # Prepare targets
            card_targets.append(action.get('card_index', 0))
            row_targets.append(action.get('row_index', 0))
            
            # Calculate value target using reward and next state value
            if done:
                value_target = reward
            else:
                next_pred = self.target_model(np.expand_dims(next_state, 0))
                value_target = reward + 0.95 * next_pred['value'][0, 0]  # Discount factor
            
            value_targets.append(value_target)
        
        states = np.array(states)
        card_targets = np.array(card_targets)
        row_targets = np.array(row_targets)
        value_targets = np.array(value_targets)
        
        # Train the model
        total_loss, card_loss, row_loss, value_loss = self.train_step(
            states, card_targets, row_targets, value_targets
        )
        
        return {
            'total_loss': float(total_loss),
            'card_loss': float(card_loss),
            'row_loss': float(row_loss),
            'value_loss': float(value_loss)
        }
    
    def update_target_model(self):
        """Update target model with current model weights."""
        self.target_model.set_weights(self.model.get_weights())

class TournamentTrainer:
    """Trainer that uses tournament results to improve models."""
    
    def __init__(self, players: List[Take6Player]):
        self.players = players
        self.trainers = [SelfPlayTrainer(player.model) for player in players]
        self.training_history = []
    
    def collect_game_data(self, game_log: Dict) -> List[Tuple]:
        """Extract training data from a game log."""
        training_data = []
        
        for round_info in game_log['rounds']:
            # For each player's action in this round
            for player_idx, (card_number, row_choice) in round_info['actions'].items():
                # Calculate reward based on penalty points gained this round
                penalty_gained = round_info['results'][player_idx][0]
                reward = -penalty_gained  # Negative penalty as reward
                
                # Store the state-action-reward tuple
                # Note: We would need to reconstruct the game state at this point
                # For now, we'll use a simplified approach
                training_data.append({
                    'player_id': player_idx,
                    'card_played': card_number,
                    'row_chosen': row_choice,
                    'reward': reward,
                    'round': round_info['round']
                })
        
        return training_data
    
    def train_from_tournament(self, tournament_results: List[Dict], 
                            training_epochs: int = 5):
        """Train models based on tournament results."""
        print(f"Training models from {len(tournament_results)} games...")
        
        all_losses = {i: [] for i in range(len(self.players))}
        
        for game_log in tournament_results:
            training_data = self.collect_game_data(game_log)
            
            # For each player involved in the game
            for player_idx in game_log['players']:
                trainer = self.trainers[player_idx]
                
                # Train multiple epochs on this game
                for epoch in range(training_epochs):
                    losses = trainer.train_from_memory()
                    if losses:
                        all_losses[player_idx].append(losses)
        
        # Update target models periodically
        for trainer in self.trainers:
            trainer.update_target_model()
        
        self.training_history.append(all_losses)
        
        return all_losses

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
        # This is a simplified check - in practice, you'd track Elo history
        return player.elo_rating < 1400  # Below average
    
    def adapt_player_strategy(self, player: Take6Player):
        """Adapt a player's strategy (e.g., increase exploration)."""
        if player.epsilon < 0.3:  # Increase exploration if too low
            player.epsilon = min(0.3, player.epsilon + 0.05)
            print(f"Increased exploration for player {player.player_id} to {player.epsilon:.3f}")
    
    def run_adaptive_cycle(self, num_games: int = 100, train_after_games: bool = True):
        """Run one cycle of adaptive training."""
        print(f"Running adaptive training cycle with {num_games} games...")
        
        # Run tournament games
        results = self.tournament.run_random_games(num_games, verbose=False)
        
        # Train models based on results
        if train_after_games:
            self.trainer.train_from_tournament(results)
        
        # Adapt struggling players
        for player in self.players:
            if self.should_adapt_player(player):
                self.adapt_player_strategy(player)
        
        return results
    
    def save_models(self, directory: str):
        """Save all player models."""
        os.makedirs(directory, exist_ok=True)
        
        for i, player in enumerate(self.players):
            model_path = os.path.join(directory, f"model_{i}")
            player.model.save_weights(model_path)
        
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
        
        with open(os.path.join(directory, 'player_stats.json'), 'w') as f:
            import json
            json.dump(stats, f, indent=2)
    
    def load_models(self, directory: str):
        """Load player models and statistics."""
        import json
        
        # Load player statistics
        stats_path = os.path.join(directory, 'player_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            for i, player in enumerate(self.players):
                if i < len(stats['player_stats']):
                    player_stats = stats['player_stats'][i]
                    player.elo_rating = player_stats['elo']
                    player.games_played = player_stats['games']
                    player.total_score = player_stats['total_score']
                    player.epsilon = player_stats['epsilon']
        
        # Load model weights
        for i, player in enumerate(self.players):
            model_path = os.path.join(directory, f"model_{i}")
            if os.path.exists(model_path + '.index'):
                try:
                    player.model.load_weights(model_path)
                    print(f"Loaded weights for player {i}")
                except Exception as e:
                    print(f"Failed to load weights for player {i}: {e}")
