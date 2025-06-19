"""
Tournament System with Elo Rating for Take 6
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import os
from collections import defaultdict
from tqdm import tqdm

from game.take6 import Take6Game, GameState
from models.neural_network import Take6Player


class EloSystem:
    """Elo rating system for tournament players."""

    def __init__(self, k_factor: int = 32, initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, players: List[Take6Player], game_results: List[int], target_penalty: int = 100):
        """
        Update Elo ratings based on game results.
        game_results: List of penalty points for each player (lower is better)
        target_penalty: Target penalty points for the match (used for performance scaling)
        """
        num_players = len(players)

        # Convert penalty points to rankings (0 = best, n-1 = worst)
        rankings = self._penalty_points_to_rankings(game_results)

        # Calculate performance multiplier based on overall game quality
        best_penalty = min(game_results)
        performance_multiplier = self._calculate_performance_multiplier(best_penalty, target_penalty)

        # Update each player's rating
        for i, player in enumerate(players):
            # Calculate expected score against all other players
            expected_total = 0.0
            actual_total = 0.0

            for j, other_player in enumerate(players):
                if i != j:
                    expected_total += self.expected_score(player.elo_rating, other_player.elo_rating)

                    # Actual score: 1 if we beat them, 0.5 if tie, 0 if they beat us
                    if rankings[i] < rankings[j]:  # We did better (lower penalty)
                        actual_total += 1.0
                    elif rankings[i] == rankings[j]:  # Tie
                        actual_total += 0.5
                    # else: actual_total += 0.0 (we did worse)

            # Normalize by number of opponents
            expected_score = expected_total / (num_players - 1)
            actual_score = actual_total / (num_players - 1)

            # Base K factor
            k = self.k_factor if player.games_played < 30 else self.k_factor // 2

            # Apply performance multiplier based on overall game quality
            k *= performance_multiplier

            player.elo_rating += k * (actual_score - expected_score)

            # Update player statistics
            player.add_game_score(game_results[i])

    def _calculate_performance_multiplier(self, best_penalty: int, target_penalty: int) -> float:
        """
        Calculate a multiplier for Elo changes based on best performer's penalty.
        Returns a value between 0.3 and 1.5:
        - 1.5 for excellent performance (very low penalty)
        - 1.0 for average performance
        - 0.3 for poor performance (close to target penalty)
        """
        if best_penalty >= target_penalty:
            # Even the best player reached target - minimal Elo changes
            return 0.1

        # Calculate ratio of best penalty to target
        penalty_ratio = best_penalty / target_penalty

        if penalty_ratio >= 0.8:  # 80-99% of target - poor performance overall
            return 0.3
        elif penalty_ratio >= 0.6:  # 60-79% of target - decent performance
            return 0.6
        elif penalty_ratio >= 0.4:  # 40-59% of target - good performance
            return 0.9
        elif penalty_ratio >= 0.2:  # 20-39% of target - very good performance
            return 1.2
        else:  # < 20% of target - excellent performance
            return 1.5

    def _penalty_points_to_rankings(self, penalty_points: List[int]) -> List[int]:
        """Convert penalty points to rankings (0 = best)."""
        # Sort indices by penalty points (ascending)
        sorted_indices = sorted(range(len(penalty_points)), key=lambda i: penalty_points[i])

        # Assign rankings
        rankings = [0] * len(penalty_points)
        for rank, player_idx in enumerate(sorted_indices):
            rankings[player_idx] = rank

        return rankings


class Tournament:
    """Tournament system for Take 6 players."""

    def __init__(self, players: List[Take6Player], elo_system: Optional[EloSystem] = None):
        self.players = players
        self.elo_system = elo_system or EloSystem()
        self.tournament_history = []
        self.round_number = 0

    def run_single_match(
        self, game_players: List[Take6Player], min_players: int = 2, max_players: int = 6, target_penalty: int = 100, verbose: bool = False
    ) -> Dict:
        """
        Run a single match with the given players until someone reaches target penalty points.
        """
        num_players = len(game_players)
        if num_players < min_players or num_players > max_players:
            raise ValueError(f"Invalid number of players: {num_players}. Must be between {min_players} and {max_players}")

        game = Take6Game(num_players=num_players, target_penalty=target_penalty)

        match_log = {
            "players": [p.player_id for p in game_players],
            "initial_elo": [p.elo_rating for p in game_players],
            "games": [],
            "target_penalty": target_penalty,
        }

        # Track cumulative penalties across games in this match
        cumulative_penalties = [0] * num_players
        game_count = 0
        max_games = 50  # Safety limit to prevent infinite loops

        # Keep playing games until someone reaches the target penalty
        while game_count < max_games:
            state = game.reset()
            game_count += 1

            game_log = {"game_number": game_count, "rounds": [], "penalty_totals_start": cumulative_penalties.copy()}

            # Play rounds until all cards are used (one complete hand)
            for round_num in range(10):
                if state.is_game_over():
                    break

                player_cards = {}

                # Get action from each player
                for i, player in enumerate(game_players):

                    valid_cards = game.get_valid_cards(i)
                    if valid_cards:  # Only if player has cards
                        player_cards[i] = player.get_card(state, valid_cards, game_player_id=i, training=True)

                if not player_cards:  # No more cards to play
                    break

                # Execute round
                round_results = game.play_round(player_cards)

                round_log = {
                    "round": round_num,
                    "cards": {int(i): int(card.number) for i, card in player_cards.items()},
                    "results": {int(i): (int(penalty), [int(c.number) for c in cards]) for i, (penalty, cards) in round_results.items()},
                    "penalty_totals": [int(cumulative_penalties[i] + state.players_penalty_points[i]) for i in range(num_players)],
                }
                game_log["rounds"].append(round_log)

                # if verbose:
                #    print(f"Round {round_num + 1}: Penalty totals: {state.players_penalty_points}")

            # Add this game's penalties to cumulative total
            for i in range(num_players):
                cumulative_penalties[i] += state.players_penalty_points[i]

            game_log["penalty_totals_end"] = cumulative_penalties.copy()
            match_log["games"].append(game_log)

            # Check if anyone has reached the target penalty (using cumulative)
            if any(penalty >= target_penalty for penalty in cumulative_penalties):
                break

            if verbose:
                penalty_map = {game_players[i].player_id: cumulative_penalties[i] for i in range(num_players)}
                print(f"Game {game_count} completed. Penalty totals: {penalty_map}")

        # If we hit the game limit, end the match anyway
        if game_count >= max_games:
            if verbose:
                print(f"Match ended after {max_games} games (safety limit)")

        # Store initial Elo ratings for comparison
        initial_elos = [p.elo_rating for p in game_players]

        # Update Elo ratings based on final match result
        final_penalties = cumulative_penalties
        self.elo_system.update_ratings(game_players, final_penalties, target_penalty)

        match_log["final_scores"] = [int(x) for x in final_penalties]
        match_log["final_elo"] = [float(p.elo_rating) for p in game_players]
        match_log["best_performer"] = int(np.argmin(final_penalties))  # Best performer has lowest penalty
        match_log["worst_performer"] = int(np.argmax(final_penalties))  # Worst performer has highest penalty
        match_log["total_games"] = game_count

        if verbose:
            # Create rankings (0 = best, higher = worse)
            penalty_rankings = self.elo_system._penalty_points_to_rankings(final_penalties)

            final_penalty_map = {game_players[i].player_id: final_penalties[i] for i in range(num_players)}
            best_performer_id = game_players[match_log["best_performer"]].player_id
            worst_performer_id = game_players[match_log["worst_performer"]].player_id
            best_penalty = min(final_penalties)
            worst_penalty = max(final_penalties)
            performance_multiplier = self.elo_system._calculate_performance_multiplier(best_penalty, target_penalty)

            print(f"Match completed after {game_count} games.")
            print(f"Final penalty scores: {final_penalty_map}")
            print(f"Best performer: Player ID {best_performer_id} (penalty: {best_penalty})")
            print(f"Worst performer: Player ID {worst_performer_id} (penalty: {worst_penalty})")
            print(
                f"Performance multiplier: {performance_multiplier:.2f} ({'excellent game' if performance_multiplier >= 1.2 else 'good game' if performance_multiplier >= 0.9 else 'decent game' if performance_multiplier >= 0.6 else 'tough game'})"
            )

            # Show Elo rating changes for each player with their ranking
            print(f"Elo rating updates:")
            for i, player in enumerate(game_players):
                elo_change = player.elo_rating - initial_elos[i]
                rank = penalty_rankings[i]
                if rank == 0:
                    status = "🥇 Best"
                elif rank == len(game_players) - 1:
                    status = "🥉 Worst"
                else:
                    status = f"🥈 #{rank + 1}"
                print(f"  Player ID {player.player_id} {status}: {initial_elos[i]:.1f} -> {player.elo_rating:.1f} ({elo_change:+.1f})")

        return match_log

    def run_round_robin(self, matches_per_matchup: int = 1, target_penalty: int = 100, players_per_match: int = 4, verbose: bool = False) -> List[Dict]:
        """Run a round-robin tournament where every combination of players competes."""
        results = []

        # Generate all combinations of players for matches
        from itertools import combinations

        matchups = list(combinations(self.players, players_per_match))

        if verbose:
            print(f"Running round-robin with {len(matchups)} matchups, {matches_per_matchup} matches each")

        # Run matches for each matchup
        for matchup in tqdm(matchups, desc="Round-Robin Progress"):
            for match_num in range(matches_per_matchup):
                match_result = self.run_single_match(list(matchup), players_per_match, players_per_match, target_penalty, verbose)
                match_result["matchup_id"] = len(results)
                match_result["match_in_matchup"] = match_num
                results.append(match_result)

        self.tournament_history.extend(results)
        self.round_number += 1

        return results

    def run_random_matches(self, num_matches: int, target_penalty: int = 100, min_players: int = 2, max_players: int = 6, verbose: bool = False) -> List[Dict]:
        """Run random matches with randomly selected players and variable match sizes."""
        results = []

        if verbose:
            print(f"Running {num_matches} random matches (target: {target_penalty} penalty points)")

        for match_num in tqdm(range(num_matches), desc="Tournament Matches"):
            # Randomly choose number of players for this match
            num_players = random.randint(min_players, max_players)

            # Randomly select players
            match_players = random.sample(self.players, num_players)

            match_result = self.run_single_match(match_players, min_players, max_players, target_penalty, verbose)
            match_result["match_id"] = match_num
            results.append(match_result)

        self.tournament_history.extend(results)

        return results

    def get_leaderboard(self) -> List[Tuple[int, float, int, float]]:
        """Get current leaderboard sorted by Elo rating."""
        leaderboard = []

        for player in self.players:
            leaderboard.append((int(player.player_id), float(player.elo_rating), int(player.games_played), float(player.get_average_score())))

        # Sort by Elo rating (descending)
        leaderboard.sort(key=lambda x: x[1], reverse=True)

        return leaderboard

    def save_results(self, filename: str):
        """Save tournament results to file."""

        def convert_to_serializable(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if hasattr(obj, "tolist"):  # numpy arrays
                return obj.tolist()
            elif hasattr(obj, "item"):  # numpy scalars
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(item) for item in obj)
            else:
                return obj

        results = {
            "tournament_history": convert_to_serializable(self.tournament_history),
            "final_leaderboard": convert_to_serializable(self.get_leaderboard()),
            "round_number": int(self.round_number),
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    def print_leaderboard(self, top_n: int = 10):
        """Print current leaderboard."""
        leaderboard = self.get_leaderboard()

        print(f"\n{'Rank':<6} {'Player':<8} {'Elo':<8} {'Games':<8} {'Avg Score':<10}")
        print("-" * 50)

        for i, (player_id, elo, games, avg_score) in enumerate(leaderboard[:top_n]):
            print(f"{i+1:<6} {player_id:<8} {elo:<8.1f} {games:<8} {avg_score:<10.2f}")


class EvolutionaryTournament(Tournament):
    """Tournament with evolutionary selection and mutation."""

    def __init__(self, players: List[Take6Player], elo_system: Optional[EloSystem] = None, selection_ratio: float = 0.3, mutation_rate: float = 0.1):
        super().__init__(players, elo_system)
        self.selection_ratio = selection_ratio
        self.mutation_rate = mutation_rate
        self.generation = 0

    # Inherit run_random_matches from parent Tournament class

    def evolve_population(self):
        """Evolve the population based on current Elo ratings."""
        from models.neural_network import ModelFactory

        # Sort players by Elo rating
        sorted_players = sorted(self.players, key=lambda p: p.elo_rating, reverse=True)

        # Select top performers
        num_survivors = int(len(self.players) * self.selection_ratio)
        survivors = sorted_players[:num_survivors]

        print(f"Generation {self.generation}: Top {num_survivors} players survive")
        print(f"Best Elo: {survivors[0].elo_rating:.1f}, Worst surviving: {survivors[-1].elo_rating:.1f}")

        # Create new population
        new_players = []

        # Keep survivors (with some Elo decay to prevent stagnation)
        for i, survivor in enumerate(survivors):
            # Reset some stats but keep Elo rating
            new_player = Take6Player(survivor.model, i, survivor.epsilon)
            new_player.elo_rating = survivor.elo_rating * 0.95  # Slight decay
            new_player.games_played = 0
            new_player.total_score = 0
            new_players.append(new_player)

        # Create offspring through mutation and crossover
        while len(new_players) < len(self.players):
            if len(survivors) >= 2 and random.random() < 0.3:
                # Crossover between two random survivors
                parent1, parent2 = random.sample(survivors, 2)
                child_model = ModelFactory.crossover_models(parent1.model, parent2.model, len(new_players))
                # Child inherits average characteristics
                epsilon = (parent1.epsilon + parent2.epsilon) / 2
                elo = (parent1.elo_rating + parent2.elo_rating) / 2 * 0.8
            else:
                # Mutation of single parent
                parent = random.choice(survivors)
                child_model = ModelFactory.mutate_model(parent.model, self.mutation_rate)
                epsilon = parent.epsilon * (1 + random.uniform(-0.1, 0.1))  # Small variation
                elo = parent.elo_rating * 0.8  # Offspring start lower

            new_player = Take6Player(child_model, len(new_players), epsilon)
            new_player.elo_rating = elo
            new_players.append(new_player)

        self.players = new_players
        self.generation += 1

        print(f"New generation created with {len(new_players)} players")
        return new_players
