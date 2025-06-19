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

    def place_card(self, player_id: int, card: Card) -> Tuple[int, List[Card]]:
        """
        Place a card and return (penalty_points_gained, cards_taken).
        """
        valid_rows = self.get_valid_rows_for_card(card)
        cards_taken = []
        penalty_points = 0

        if self.must_take_row(card):
            # Player must take a row
            # Choose row with minimum penalty points
            chosen_row = min(range(4), key=lambda i: sum(c.penalty_points for c in self.rows[i]))

            cards_taken = self.rows[chosen_row].copy()
            penalty_points = sum(c.penalty_points for c in cards_taken)
            self.rows[chosen_row] = [card]
        else:
            # Choose the row where the last card is closest to our card
            best_row = min(valid_rows, key=lambda i: card.number - self.rows[i][-1].number)
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

        # Penalty points for all players (normalized) - always use 6 slots for consistency
        penalty_vector = np.zeros(6, dtype=float)  # Always 6 slots for max players
        for i in range(min(len(self.players_penalty_points), 6)):
            penalty_vector[i] = self.players_penalty_points[i] / 100.0  # Normalize
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

    def __init__(self, num_players: int = 4, target_penalty: int = 100):
        self.num_players = num_players
        self.target_penalty = target_penalty
        self.reset()

    def reset(self) -> GameState:
        """Reset game to initial state."""
        self.state = GameState(self.num_players)
        self.state.deal_cards()
        return self.state

    def is_game_finished(self) -> bool:
        """Check if any player has reached the target penalty points."""
        return any(penalty >= self.target_penalty for penalty in self.state.players_penalty_points)

    def play_complete_game(self) -> Dict:
        """Play a complete game until someone reaches target penalty points."""
        game_log = {"rounds": [], "final_scores": [], "winner": -1, "total_rounds": 0}

        round_count = 0

        while not self.is_game_finished():
            # Check if we need to deal new cards (start new hand)
            if self.state.is_game_over():
                self.state.deal_cards()
                round_count = 0

            # Play one round (all 10 cards from current hand)
            for round_in_hand in range(10):
                if self.is_game_finished():
                    break

                round_count += 1
                # This would need player actions - we'll handle this in the tournament
                break

            # If we completed a hand without anyone reaching target, continue
            if not self.is_game_finished() and self.state.is_game_over():
                continue
            else:
                break

        game_log["final_scores"] = self.state.players_penalty_points.copy()
        game_log["winner"] = self.get_winner_by_lowest_penalty()
        game_log["total_rounds"] = round_count

        return game_log

    def get_winner_by_lowest_penalty(self) -> int:
        """Get winner as player with lowest penalty points when game ends."""
        return int(np.argmin(self.state.players_penalty_points))

    def play_round(self, player_cards: Dict[int, Card]) -> Dict[int, Tuple[int, List[Card]]]:
        """
        Play one round with all player actions.
        player_actions: {player_id: card_to_play}
        Returns: {player_id: (penalty_points_gained, cards_taken)}
        """
        self.state.current_round_cards = {}
        results = {}

        # Collect all played cards
        for player_id, card in player_cards.items():
            self.state.current_round_cards[player_id] = card
            # Remove card from player's hand
            self.state.players_hands[player_id].remove(card)

        # Sort players by card number (lowest first)
        sorted_players = sorted(player_cards.keys(), key=lambda p: player_cards[p].number)

        # Execute moves in order
        for player_id in sorted_players:
            card = player_cards[player_id]
            penalty_points, cards_taken = self.state.place_card(player_id, card)
            results[player_id] = (penalty_points, cards_taken)

        self.state.round_number += 1
        return results

    def get_valid_cards(self, player_id: int) -> List[Card]:
        return list(self.state.players_hands[player_id])

    def clone_state(self) -> "GameState":
        """Create a deep copy of the current game state."""
        new_state = GameState(self.num_players)
        new_state.rows = [row.copy() for row in self.state.rows]
        new_state.players_hands = [hand.copy() for hand in self.state.players_hands]
        new_state.players_penalty_points = self.state.players_penalty_points.copy()
        new_state.current_round_cards = self.state.current_round_cards.copy()
        new_state.round_number = self.state.round_number
        return new_state
