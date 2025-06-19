"""
Neural Network Models for Take 6 Players
"""

import os
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Optional
import random
from game.take6 import Card, GameState, Take6Game

# Configure basic TensorFlow settings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN warnings


def configure_gpu():
    """Configure GPU settings for optimal performance."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU configuration successful. Using {len(gpus)} GPU(s) for neural network training.")
            return True
        else:
            print("No GPU devices found, using CPU.")
            return False
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        return False


class Take6Network(tf.keras.Model):
    """Neural network for playing Take 6."""

    def __init__(self, input_size: int = 2608, hidden_size: int = 512, name: str = "Take6Network"):
        """
        Initialize the network.
        input_size: Size of game state vector (default calculated from game state)
        - Player hand: 104 bits
        - 4 rows * 6 positions * 104 cards: 2496 bits
        - 6 penalty points (normalized): 6 floats (fixed size for up to 6 players)
        - Round number (normalized): 1 float
        Total: 104 + 2496 + 6 + 1 = 2607, but we'll use 2608 for alignment
        """
        super().__init__(name=name)

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Feature extraction layers
        self.feature_dense1 = tf.keras.layers.Dense(hidden_size, activation="relu", name="feature_1")
        self.feature_dropout1 = tf.keras.layers.Dropout(0.2)
        self.feature_dense2 = tf.keras.layers.Dense(hidden_size // 2, activation="relu", name="feature_2")
        self.feature_dropout2 = tf.keras.layers.Dropout(0.2)

        # Card selection head
        self.card_dense1 = tf.keras.layers.Dense(hidden_size // 4, activation="relu", name="card_1")
        self.card_output = tf.keras.layers.Dense(104, activation="softmax", name="card_output")  # Probability for each card

        # Value estimation head
        self.value_dense1 = tf.keras.layers.Dense(64, activation="relu", name="value_1")
        self.value_output = tf.keras.layers.Dense(1, activation="tanh", name="value_output")  # Expected score

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

        # Value estimation
        value_features = self.value_dense1(features)
        value = self.value_output(value_features)

        return {"card_probs": card_probs, "value": value}


class Take6Player:
    """AI player using neural network for Take 6."""

    def __init__(self, model: Take6Network, player_id: int, epsilon: float = 0.1):
        self.model = model
        self.player_id = player_id
        self.epsilon = epsilon  # Exploration rate
        self.elo_rating = 1500.0  # Starting Elo rating
        self.games_played = 0
        self.total_score = 0

    def get_card(
        self, game_state: GameState, valid_cards: List[Card], game_player_id: Optional[int] = None, training: bool = False
    ) -> Card:
        """
        Choose an action based on current game state.
        Returns: (card_to_play, row_to_take_if_needed)
        """
        # Use game_player_id if provided, otherwise use self.player_id
        player_id_for_state = game_player_id if game_player_id is not None else self.player_id

        # DEBUG: Render the current game state
        #self._debug_render_game_state(game_state, player_id_for_state, valid_cards)

        if random.random() < self.epsilon and training:
            print(f"Player {self.player_id} (Elo: {self.elo_rating:.2f}) - Random exploration")
            # Random exploration
            return self._random_card(valid_cards)

        # Get neural network prediction
        state_vector = game_state.get_game_state_vector(player_id_for_state)
        state_vector = np.expand_dims(state_vector, axis=0)  # Add batch dimension

        # Print non-zero elements of the state vector, scaled by 1000 and rounded to two decimals
        #nonzero_map = {i: round(float(v) * 1000, 2) for i, v in enumerate(state_vector[0]) if v != 0}
        #print(f"Player {self.player_id} (Elo: {self.elo_rating:.2f}) - Non-zero state vector entries (x1000): {nonzero_map}")
        predictions = self.model(state_vector, training=training)

        # Choose card based on network output and valid actions
        card_probs = predictions["card_probs"][0].numpy()

        # Filter to only valid cards and choose best one
        valid_card_indices = [card.number - 1 for card in valid_cards]

        # Mask invalid cards
        masked_card_probs = np.zeros_like(card_probs)
        masked_card_probs[valid_card_indices] = card_probs[valid_card_indices]

        #nonzero_probs = {idx + 1: round(float(prob) * 1000, 2) for idx, prob in enumerate(masked_card_probs) if prob > 0}
        #print(f"Player {self.player_id} (Elo: {self.elo_rating:.2f}) - Card probabilities (non-zero, x1000): {nonzero_probs}")

        # Print only non-zero probabilities for valid cards, scaled by 1000 and rounded to two decimals
        if np.sum(masked_card_probs) > 0:
            chosen_card_idx = np.argmax(masked_card_probs)
            chosen_card = None
            for card in valid_cards:
                if card.number - 1 == chosen_card_idx:
                    chosen_card = card
                    break
        else:
            # Fallback to random if no valid probabilities
            print("Warning: No valid card probabilities found, falling back to random choice.")
            chosen_card = random.choice(valid_cards)

        return chosen_card

    def _random_card(self, valid_cards: List[Card]) -> Card:
        return random.choice(valid_cards)

    def update_elo(self, score_rank: int, num_players: int):
        """
        Update Elo rating based on game result.
        score_rank: 0 = best (lowest penalty points), num_players-1 = worst
        """
        # Calculate expected score based on average opponent rating
        # For simplicity, assume average opponent rating is 1500
        expected_score = 1.0 / (1.0 + 10 ** ((1500 - self.elo_rating) / 400))

        # Actual score: 1.0 for 1st place, 0.5 for 2nd, 0.0 for last, etc.
        actual_score = (num_players - 1 - score_rank) / (num_players - 1)

        # Update rating
        k_factor = 32 if self.games_played < 30 else 16  # Higher K for new players
        self.elo_rating += k_factor * (actual_score - expected_score)

        self.games_played += 1

    def add_game_score(self, penalty_points: int):
        """Add penalty points from a game to total score."""
        self.total_score += penalty_points
        self.games_played += 1  # Increment games played counter

    def get_average_score(self) -> float:
        """Get average penalty points per game."""
        return self.total_score / max(1, self.games_played)

    def _debug_render_game_state(self, game_state: GameState, player_id: int, valid_cards: List[Card]):
        """Debug function to render the current game state."""
        print(f"\n{'='*60}")
        print(f"DEBUG RENDER - Player {player_id} Action Selection")
        print(f"{'='*60}")

        # Display rows
        print("Current Board Rows:")
        for i, row in enumerate(game_state.rows):
            if row:
                row_str = " -> ".join([f"{card.number}" for card in row])
                penalty_sum = sum(card.penalty_points for card in row)
                print(f"  Row {i}[{penalty_sum}]: {row_str}")
            else:
                print(f"  Row {i}: [Empty]")

        # Display player's hand
        player_hand = game_state.players_hands[player_id]
        hand_str = ", ".join([f"{card.number}" for card in sorted(player_hand, key=lambda c: c.number)])
        print(f"\nPlayer {player_id} Hand: [{hand_str}]")

        # Display penalty points
        print(f"\nCurrent Penalty Points:")
        for i, penalty in enumerate(game_state.players_penalty_points):
            marker = " <-- YOU" if i == player_id else ""
            print(f"  Player {i}: {penalty} points{marker}")

        print(f"Round: {game_state.round_number}")
        print(f"{'='*60}\n")


class ModelFactory:
    """Factory for creating and managing neural network models."""

    @staticmethod
    def create_model(model_id: int, input_size: int = 2608, hidden_size_base: int = 512, diversity_factor: float = 0.3) -> Take6Network:
        """Create a new model with some architectural diversity."""
        # Add some variation to hidden layer sizes
        hidden_size = int(hidden_size_base * (1.0 + diversity_factor * (random.random() - 0.5)))
        hidden_size = max(256, min(1024, hidden_size))  # Keep within reasonable bounds

        model = Take6Network(input_size=input_size, hidden_size=hidden_size, name=f"Model_{model_id}")

        # Add some random initialization variation
        for layer in model.layers:
            if hasattr(layer, "kernel_initializer"):
                # Slightly vary the initialization scale
                scale_factor = 1.0 + 0.2 * (random.random() - 0.5)
                if hasattr(layer, "kernel"):
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
    def create_population(population_size: int, input_size: int = 2608) -> List[Take6Player]:
        """Create a population of players with diverse neural networks."""
        # Configure GPU on first model creation
        configure_gpu()

        players = []

        for i in range(population_size):
            # Create models with SAME architecture for crossover compatibility
            # Only vary initialization, not architecture
            model = ModelFactory.create_model(i, input_size, diversity_factor=0.0)

            # Add variation in exploration rates (epsilon)
            epsilon = 0.05 + 0.15 * (i / population_size)  # Range from 0.05 to 0.20

            # Add small variation in starting Elo ratings to encourage diversity
            elo_variation = random.uniform(-50, 50)

            player = Take6Player(model, i, epsilon)
            player.elo_rating += elo_variation
            players.append(player)

        return players

    @staticmethod
    def mutate_model(parent_model: Take6Network, mutation_rate: float = 0.1, mutation_strength: float = 0.02) -> Take6Network:
        """Create a mutated copy of a model with more sophisticated mutations."""
        # Ensure parent model is built
        if not parent_model.built:
            dummy_input = tf.random.normal((1, parent_model.input_size))
            _ = parent_model(dummy_input)

        new_model = Take6Network(parent_model.input_size, parent_model.hidden_size)

        # Build the model by calling it with dummy data
        dummy_input = tf.random.normal((1, parent_model.input_size))
        _ = new_model(dummy_input)

        # Copy weights from parent
        parent_weights = parent_model.get_weights()
        new_weights = []

        for weight_matrix in parent_weights:
            new_weight = weight_matrix.copy()

            # Apply mutations
            if random.random() < mutation_rate:
                # Different types of mutations
                mutation_type = random.choice(["gaussian", "dropout", "scaling"])

                if mutation_type == "gaussian":
                    # Gaussian noise mutation
                    noise = np.random.normal(0, mutation_strength, weight_matrix.shape)
                    new_weight += noise
                elif mutation_type == "dropout":
                    # Random dropout of some weights
                    mask = np.random.random(weight_matrix.shape) > 0.1
                    new_weight *= mask
                elif mutation_type == "scaling":
                    # Scale some weights
                    scale = 1.0 + random.uniform(-0.1, 0.1)
                    new_weight *= scale

            new_weights.append(new_weight)

        new_model.set_weights(new_weights)
        return new_model

    @staticmethod
    def crossover_models(parent1: Take6Network, parent2: Take6Network, child_id: int) -> Take6Network:
        """Create a child model by crossing over two parent models."""
        # Ensure parents have compatible architectures
        if parent1.input_size != parent2.input_size or parent1.hidden_size != parent2.hidden_size:
            print(f"Warning: Parent models have incompatible architectures. Using mutation instead.")
            return ModelFactory.mutate_model(parent1, 0.15)

        child_model = Take6Network(parent1.input_size, parent1.hidden_size, name=f"Child_{child_id}")

        # Build the model by calling it with dummy data
        dummy_input = tf.random.normal((1, parent1.input_size))
        _ = child_model(dummy_input)

        parent1_weights = parent1.get_weights()
        parent2_weights = parent2.get_weights()

        # Additional safety check for weight shapes
        if len(parent1_weights) != len(parent2_weights):
            print(f"Warning: Parent models have different numbers of layers. Using mutation instead.")
            return ModelFactory.mutate_model(parent1, 0.15)

        child_weights = []

        for i, (w1, w2) in enumerate(zip(parent1_weights, parent2_weights)):
            # Check shape compatibility
            if w1.shape != w2.shape:
                print(f"Warning: Weight matrices at layer {i} have incompatible shapes: {w1.shape} vs {w2.shape}. Using mutation instead.")
                return ModelFactory.mutate_model(parent1, 0.15)

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
