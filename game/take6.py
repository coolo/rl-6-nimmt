"""
Take 6 (6 nimmt!) Game Implementation
Core game logic for the card game.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
import random

class Card:
    """Represents a card with a number and penalty points."""
    
    def __init__(self, number: int):
        self.number = number
        self.penalty_points = self._calculate_penalty_points(number)
    
    def _calculate_penalty_points(self, number: int) -> int:
        """Calculate penalty points based on card number."""
        if number == 55:
            return 7
        elif number % 11 == 0:
            return 5
        elif number % 10 == 0:
            return 3
        elif number % 5 == 0:
            return 2
        else:
            return 1
    
    def __str__(self):
        return f"Card({self.number}, {self.penalty_points}pts)"
    
    def __repr__(self):
        return self.__str__()

class GameState:
    """Represents the current state of a Take 6 game."""
    
    def __init__(self, num_players: int = 4):
        self.num_players = num_players
        self.deck = [Card(i) for i in range(1, 105)]  # Cards 1-104
        self.rows = [[] for _ in range(4)]  # 4 rows
        self.players_hands = [[] for _ in range(num_players)]
        self.players_penalty_points = [0] * num_players
        self.current_round_cards = {}  # player_id -> card
        self.round_number = 0
        
    def deal_cards(self):
        """Deal 10 cards to each player and initialize rows."""
        random.shuffle(self.deck)
        
        # Deal 10 cards to each player
        for i in range(10):
            for player in range(self.num_players):
                self.players_hands[player].append(self.deck.pop())
        
        # Initialize 4 rows with one card each
        for i in range(4):
            self.rows[i] = [self.deck.pop()]
        
        # Sort each player's hand
        for hand in self.players_hands:
            hand.sort(key=lambda card: card.number)
    
    def get_valid_rows_for_card(self, card: Card) -> List[int]:
        """Get valid rows where a card can be placed."""
        valid_rows = []
        
        for row_idx, row in enumerate(self.rows):
            if not row:  # Empty row (shouldn't happen in normal play)
                valid_rows.append(row_idx)
            elif card.number > row[-1].number:  # Card number is higher than last card in row
                valid_rows.append(row_idx)
        
        return valid_rows
    
    def must_take_row(self, card: Card) -> bool:
        """Check if player must take a row (card is smaller than all row ends)."""
        return len(self.get_valid_rows_for_card(card)) == 0
    
    def place_card(self, player_id: int, card: Card, chosen_row: Optional[int] = None) -> Tuple[int, List[Card]]:
        """
        Place a card and return (penalty_points_gained, cards_taken).
        If chosen_row is None, automatically place in the best valid row.
        """
        valid_rows = self.get_valid_rows_for_card(card)
        cards_taken = []
        penalty_points = 0
        
        if self.must_take_row(card):
            # Player must take a row
            if chosen_row is None:
                # Choose row with minimum penalty points
                chosen_row = min(range(4), key=lambda i: sum(c.penalty_points for c in self.rows[i]))
            
            cards_taken = self.rows[chosen_row].copy()
            penalty_points = sum(c.penalty_points for c in cards_taken)
            self.rows[chosen_row] = [card]
        else:
            # Place card in valid row
            if chosen_row is None:
                # Choose the row where the last card is closest to our card
                best_row = min(valid_rows, key=lambda i: card.number - self.rows[i][-1].number)
            else:
                best_row = chosen_row
            
            self.rows[best_row].append(card)
            
            # Check if row is full (6 cards)
            if len(self.rows[best_row]) == 6:
                cards_taken = self.rows[best_row][:-1]  # Take first 5 cards
                penalty_points = sum(c.penalty_points for c in cards_taken)
                self.rows[best_row] = [card]  # Keep only the new card
        
        self.players_penalty_points[player_id] += penalty_points
        return penalty_points, cards_taken
    
    def get_game_state_vector(self, player_id: int) -> np.ndarray:
        """Get game state as a vector for neural network input."""
        state = []
        
        # Player's hand (104 bits - one for each possible card)
        hand_vector = np.zeros(104)
        for card in self.players_hands[player_id]:
            hand_vector[card.number - 1] = 1
        state.extend(hand_vector.tolist())
        
        # Current rows state (4 rows * 6 positions * 104 possible cards)
        for row in self.rows:
            row_vector = np.zeros(6 * 104)
            for pos, card in enumerate(row):
                if pos < 6:  # Safety check
                    row_vector[pos * 104 + card.number - 1] = 1
            state.extend(row_vector.tolist())
        
        # Penalty points for all players (normalized)
        penalty_vector = np.array(self.players_penalty_points, dtype=float) / 100.0  # Normalize
        state.extend(penalty_vector.tolist())
        
        # Round number (normalized)
        state.append(float(self.round_number) / 10.0)
        
        return np.array(state, dtype=np.float32)
    
    def is_game_over(self) -> bool:
        """Check if game is over (all cards played)."""
        return all(len(hand) == 0 for hand in self.players_hands)
    
    def get_winner(self) -> int:
        """Get winner (player with minimum penalty points)."""
        return np.argmin(self.players_penalty_points)

class Take6Game:
    """Main game controller for Take 6."""
    
    def __init__(self, num_players: int = 4):
        self.num_players = num_players
        self.reset()
    
    def reset(self) -> GameState:
        """Reset game to initial state."""
        self.state = GameState(self.num_players)
        self.state.deal_cards()
        return self.state
    
    def play_round(self, player_actions: Dict[int, Tuple[Card, Optional[int]]]) -> Dict[int, Tuple[int, List[Card]]]:
        """
        Play one round with all player actions.
        player_actions: {player_id: (card_to_play, chosen_row_if_must_take)}
        Returns: {player_id: (penalty_points_gained, cards_taken)}
        """
        self.state.current_round_cards = {}
        results = {}
        
        # Collect all played cards
        for player_id, (card, chosen_row) in player_actions.items():
            self.state.current_round_cards[player_id] = (card, chosen_row)
            # Remove card from player's hand
            self.state.players_hands[player_id].remove(card)
        
        # Sort players by card number (lowest first)
        sorted_players = sorted(player_actions.keys(), 
                              key=lambda p: player_actions[p][0].number)
        
        # Execute moves in order
        for player_id in sorted_players:
            card, chosen_row = player_actions[player_id]
            penalty_points, cards_taken = self.state.place_card(player_id, card, chosen_row)
            results[player_id] = (penalty_points, cards_taken)
        
        self.state.round_number += 1
        return results
    
    def get_valid_actions(self, player_id: int) -> List[Tuple[Card, List[int]]]:
        """
        Get valid actions for a player.
        Returns: [(card, valid_rows)] where valid_rows is empty if must take a row.
        """
        valid_actions = []
        
        for card in self.state.players_hands[player_id]:
            valid_rows = self.state.get_valid_rows_for_card(card)
            if not valid_rows:  # Must take a row
                valid_actions.append((card, list(range(4))))  # Can choose any row
            else:
                valid_actions.append((card, valid_rows))
        
        return valid_actions
    
    def play_game(self, players: List) -> Tuple[int, List[int]]:
        """
        Play a complete game with given players.
        Returns: (winner_id, final_penalty_points)
        """
        self.reset()
        
        while not self.state.is_game_over():
            # Get actions from all players
            player_actions = {}
            
            for player_id, player in enumerate(players):
                valid_actions = self.get_valid_actions(player_id)
                if valid_actions:
                    # Player chooses an action
                    chosen_card, chosen_row = player.choose_action(
                        self.state, player_id, valid_actions
                    )
                    player_actions[player_id] = (chosen_card, chosen_row)
            
            # Play the round
            round_results = self.play_round(player_actions)
            
            # Notify players of results (for learning)
            for player_id, player in enumerate(players):
                if hasattr(player, 'observe_result'):
                    penalty_gained, cards_taken = round_results.get(player_id, (0, []))
                    player.observe_result(penalty_gained, cards_taken)
        
        winner_id = self.state.get_winner()
        final_scores = self.state.players_penalty_points.copy()
        
        return winner_id, final_scores
    
    def get_game_state_for_player(self, player_id: int) -> np.ndarray:
        """Get the current game state from a specific player's perspective."""
        return self.state.get_game_state_vector(player_id)
    
    def clone_state(self) -> 'GameState':
        """Create a deep copy of the current game state."""
        new_state = GameState(self.num_players)
        new_state.rows = [row.copy() for row in self.state.rows]
        new_state.players_hands = [hand.copy() for hand in self.state.players_hands]
        new_state.players_penalty_points = self.state.players_penalty_points.copy()
        new_state.current_round_cards = self.state.current_round_cards.copy()
        new_state.round_number = self.state.round_number
        return new_state

class RandomPlayer:
    """Simple random player for testing game mechanics."""
    
    def __init__(self, player_id: int):
        self.player_id = player_id
    
    def choose_action(self, game_state: GameState, player_id: int, 
                     valid_actions: List[Tuple[Card, List[int]]]) -> Tuple[Card, Optional[int]]:
        """Choose a random valid action."""
        if not valid_actions:
            return None, None
        
        # Choose random card
        card, valid_rows = random.choice(valid_actions)
        
        # Choose random row if needed
        chosen_row = None
        if valid_rows:
            if len(valid_rows) > 1:
                chosen_row = random.choice(valid_rows)
            else:
                chosen_row = valid_rows[0] if valid_rows else None
        
        return card, chosen_row
    
    def observe_result(self, penalty_gained: int, cards_taken: List[Card]):
        """Observe the result (random player doesn't learn)."""
        pass


def print_game_state(game_state: GameState):
    """Print current game state for debugging."""
    print("=" * 50)
    print("GAME STATE")
    print("=" * 50)
    
    print(f"Round: {game_state.round_number}")
    print()
    
    print("ROWS:")
    for i, row in enumerate(game_state.rows):
        cards_str = " -> ".join([f"{card.number}({card.penalty_points})" for card in row])
        print(f"  Row {i+1}: {cards_str}")
    print()
    
    print("PLAYERS:")
    for i in range(game_state.num_players):
        hand_str = ", ".join([str(card.number) for card in game_state.players_hands[i]])
        print(f"  Player {i+1}: [{hand_str}] (Penalty: {game_state.players_penalty_points[i]})")
    print()


def test_game():
    """Simple test function to verify game mechanics."""
    print("Testing Take 6 Game Implementation")
    print("=" * 50)
    
    # Create game with 4 random players
    game = Take6Game(num_players=4)
    players = [RandomPlayer(i) for i in range(4)]
    
    # Play a game
    winner_id, final_scores = game.play_game(players)
    
    print("GAME COMPLETED!")
    print(f"Winner: Player {winner_id + 1}")
    print("Final Scores:")
    for i, score in enumerate(final_scores):
        print(f"  Player {i+1}: {score} penalty points")
    
    return winner_id, final_scores


if __name__ == "__main__":
    test_game()
