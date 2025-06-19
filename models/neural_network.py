"""
Neural Network Models for Take 6 Players
"""
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Optional
import random
from game.take6 import Card, GameState, Take6Game

class Take6Network(tf.keras.Model):
    """Neural network for playing Take 6."""
    
    def __init__(self, input_size: int = 2604, hidden_size: int = 512, name: str = "Take6Network"):
        """
        Initialize the network.
        input_size: Size of game state vector (default calculated from game state)
        - Player hand: 104 bits
        - 4 rows * 6 positions * 104 cards: 2496 bits
        - 4 penalty points (normalized): 4 floats
        - Round number (normalized): 1 float
        Total: 104 + 2496 + 4 + 1 = 2605, but we'll use 2604 for alignment
        """
        super().__init__(name=name)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Feature extraction layers
        self.feature_dense1 = tf.keras.layers.Dense(hidden_size, activation='relu', name='feature_1')
        self.feature_dropout1 = tf.keras.layers.Dropout(0.2)
        self.feature_dense2 = tf.keras.layers.Dense(hidden_size // 2, activation='relu', name='feature_2')
        self.feature_dropout2 = tf.keras.layers.Dropout(0.2)
        
        # Card selection head
        self.card_dense1 = tf.keras.layers.Dense(hidden_size // 4, activation='relu', name='card_1')
        self.card_output = tf.keras.layers.Dense(104, activation='softmax', name='card_output')  # Probability for each card
        
        # Row selection head (when must take a row)
        self.row_dense1 = tf.keras.layers.Dense(64, activation='relu', name='row_1')
        self.row_output = tf.keras.layers.Dense(4, activation='softmax', name='row_output')  # Probability for each row
        
        # Value estimation head
        self.value_dense1 = tf.keras.layers.Dense(64, activation='relu', name='value_1')
        self.value_output = tf.keras.layers.Dense(1, activation='tanh', name='value_output')  # Expected score
    
    def call(self, inputs, training=False):
        """Forward pass through the network."""
        # Feature extraction
        x = self.feature_dense1(inputs)
        x = self.feature_dropout1(x, training=training)
        x = self.feature_dense2(x)
        features = self.feature_dropout2(x, training=training)
        
        # Card selection
        card_features = self.card_dense1(features)
        card_probs = self.card_output(card_features)
        
        # Row selection
        row_features = self.row_dense1(features)
        row_probs = self.row_output(row_features)
        
        # Value estimation
        value_features = self.value_dense1(features)
        value = self.value_output(value_features)
        
        return {
            'card_probs': card_probs,
            'row_probs': row_probs,
            'value': value
        }

class Take6Player:
    """AI player using neural network for Take 6."""
    
    def __init__(self, model: Take6Network, player_id: int, epsilon: float = 0.1):
        self.model = model
        self.player_id = player_id
        self.epsilon = epsilon  # Exploration rate
        self.elo_rating = 1500.0  # Starting Elo rating
        self.games_played = 0
        self.total_score = 0
        
    def get_action(self, game_state: GameState, valid_actions: List[Tuple[Card, List[int]]], 
                   game_player_id: Optional[int] = None, training: bool = False) -> Tuple[Card, Optional[int]]:
        """
        Choose an action based on current game state.
        Returns: (card_to_play, row_to_take_if_needed)
        """
        if random.random() < self.epsilon and training:
            # Random exploration
            return self._random_action(valid_actions)
        
        # Use game_player_id if provided, otherwise use self.player_id
        player_id_for_state = game_player_id if game_player_id is not None else self.player_id
        
        # Get neural network prediction
        state_vector = game_state.get_game_state_vector(player_id_for_state)
        state_vector = np.expand_dims(state_vector, axis=0)  # Add batch dimension
        
        predictions = self.model(state_vector, training=training)
        
        # Choose card based on network output and valid actions
        card_probs = predictions['card_probs'][0].numpy()
        row_probs = predictions['row_probs'][0].numpy()
        
        # Filter to only valid cards and choose best one
        valid_cards = [card for card, _ in valid_actions]
        valid_card_indices = [card.number - 1 for card in valid_cards]
        
        # Mask invalid cards
        masked_card_probs = np.zeros_like(card_probs)
        masked_card_probs[valid_card_indices] = card_probs[valid_card_indices]
        
        if np.sum(masked_card_probs) > 0:
            masked_card_probs = masked_card_probs / np.sum(masked_card_probs)
            chosen_card_idx = np.random.choice(104, p=masked_card_probs)
            chosen_card = None
            for card in valid_cards:
                if card.number - 1 == chosen_card_idx:
                    chosen_card = card
                    break
        else:
            # Fallback to random if no valid probabilities
            chosen_card = random.choice(valid_cards)
        
        # Find the corresponding valid rows for chosen card
        chosen_row = None
        for card, valid_rows in valid_actions:
            if card == chosen_card:
                if len(valid_rows) == 4:  # Must take a row
                    # Choose row based on network output
                    chosen_row = np.argmax(row_probs)
                elif len(valid_rows) == 1:
                    chosen_row = valid_rows[0]
                else:
                    # Multiple valid rows, choose randomly among them for now
                    chosen_row = random.choice(valid_rows)
                break
        
        return chosen_card, chosen_row
    
    def _random_action(self, valid_actions: List[Tuple[Card, List[int]]]) -> Tuple[Card, Optional[int]]:
        """Choose a random valid action."""
        card, valid_rows = random.choice(valid_actions)
        
        if len(valid_rows) == 0:
            row = None
        elif len(valid_rows) == 4:  # Must take a row
            row = random.choice(valid_rows)
        else:
            row = random.choice(valid_rows) if valid_rows else None
        
        return card, row
    
    def update_elo(self, score_rank: int, num_players: int):
        """
        Update Elo rating based on game result.
        score_rank: 0 = best (lowest penalty points), num_players-1 = worst
        """
        # Calculate expected score based on average opponent rating
        # For simplicity, assume average opponent rating is 1500
        expected_score = 1.0 / (1.0 + 10**((1500 - self.elo_rating) / 400))
        
        # Actual score: 1.0 for 1st place, 0.5 for 2nd, 0.0 for last, etc.
        actual_score = (num_players - 1 - score_rank) / (num_players - 1)
        
        # Update rating
        k_factor = 32 if self.games_played < 30 else 16  # Higher K for new players
        self.elo_rating += k_factor * (actual_score - expected_score)
        
        self.games_played += 1
    
    def add_game_score(self, penalty_points: int):
        """Add penalty points from a game to total score."""
        self.total_score += penalty_points
    
    def get_average_score(self) -> float:
        """Get average penalty points per game."""
        return self.total_score / max(1, self.games_played)

class ModelFactory:
    """Factory for creating and managing neural network models."""
    
    @staticmethod
    def create_model(model_id: int, input_size: int = 2605, 
                    hidden_size_base: int = 512, diversity_factor: float = 0.3) -> Take6Network:
        """Create a new model with some architectural diversity."""
        # Add some variation to hidden layer sizes
        hidden_size = int(hidden_size_base * (1.0 + diversity_factor * (random.random() - 0.5)))
        hidden_size = max(256, min(1024, hidden_size))  # Keep within reasonable bounds
        
        model = Take6Network(input_size=input_size, hidden_size=hidden_size, name=f"Model_{model_id}")
        
        # Add some random initialization variation
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer'):
                # Slightly vary the initialization scale
                scale_factor = 1.0 + 0.2 * (random.random() - 0.5)
                if hasattr(layer, 'kernel'):
                    # Apply small random perturbation to initial weights
                    initial_weights = layer.get_weights()
                    if initial_weights:
                        perturbed_weights = []
                        for weight in initial_weights:
                            noise = np.random.normal(0, 0.01 * scale_factor, weight.shape)
                            perturbed_weights.append(weight + noise)
                        layer.set_weights(perturbed_weights)
        
        return model
    
    @staticmethod
    def create_population(population_size: int, input_size: int = 2605) -> List[Take6Player]:
        """Create a population of players with diverse neural networks."""
        players = []
        
        for i in range(population_size):
            # Create diverse models
            model = ModelFactory.create_model(i, input_size, diversity_factor=0.4)
            
            # Add variation in exploration rates (epsilon)
            epsilon = 0.05 + 0.15 * (i / population_size)  # Range from 0.05 to 0.20
            
            # Add small variation in starting Elo ratings to encourage diversity
            elo_variation = random.uniform(-50, 50)
            
            player = Take6Player(model, i, epsilon)
            player.elo_rating += elo_variation
            players.append(player)
        
        return players
    
    @staticmethod
    def mutate_model(parent_model: Take6Network, mutation_rate: float = 0.1, 
                    mutation_strength: float = 0.02) -> Take6Network:
        """Create a mutated copy of a model with more sophisticated mutations."""
        new_model = Take6Network(parent_model.input_size, parent_model.hidden_size)
        
        # Copy weights from parent
        parent_weights = parent_model.get_weights()
        new_weights = []
        
        for weight_matrix in parent_weights:
            new_weight = weight_matrix.copy()
            
            # Apply mutations
            if random.random() < mutation_rate:
                # Different types of mutations
                mutation_type = random.choice(['gaussian', 'dropout', 'scaling'])
                
                if mutation_type == 'gaussian':
                    # Gaussian noise mutation
                    noise = np.random.normal(0, mutation_strength, weight_matrix.shape)
                    new_weight += noise
                elif mutation_type == 'dropout':
                    # Random dropout of some weights
                    mask = np.random.random(weight_matrix.shape) > 0.1
                    new_weight *= mask
                elif mutation_type == 'scaling':
                    # Scale some weights
                    scale = 1.0 + random.uniform(-0.1, 0.1)
                    new_weight *= scale
            
            new_weights.append(new_weight)
        
        new_model.set_weights(new_weights)
        return new_model
    
    @staticmethod
    def crossover_models(parent1: Take6Network, parent2: Take6Network, 
                        child_id: int) -> Take6Network:
        """Create a child model by crossing over two parent models."""
        child_model = Take6Network(parent1.input_size, parent1.hidden_size, name=f"Child_{child_id}")
        
        parent1_weights = parent1.get_weights()
        parent2_weights = parent2.get_weights()
        child_weights = []
        
        for w1, w2 in zip(parent1_weights, parent2_weights):
            # Random blend of parent weights
            alpha = random.random()
            child_weight = alpha * w1 + (1 - alpha) * w2
            
            # Add small mutation
            if random.random() < 0.1:
                noise = np.random.normal(0, 0.01, child_weight.shape)
                child_weight += noise
            
            child_weights.append(child_weight)
        
        child_model.set_weights(child_weights)
        return child_model
