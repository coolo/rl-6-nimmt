"""
Tournament System with Elo Rating for Take 6
"""
import random
import numpy as np
from typing import List, Dict, Tuple
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
        return 1.0 / (1.0 + 10**((rating_b - rating_a) / 400))
    
    def update_ratings(self, players: List[Take6Player], game_results: List[int]):
        """
        Update Elo ratings based on game results.
        game_results: List of penalty points for each player (lower is better)
        """
        num_players = len(players)
        
        # Convert penalty points to rankings (0 = best, n-1 = worst)
        rankings = self._penalty_points_to_rankings(game_results)
        
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
            
            # Update rating
            k = self.k_factor if player.games_played < 30 else self.k_factor // 2
            player.elo_rating += k * (actual_score - expected_score)
            
            # Update player statistics
            player.add_game_score(game_results[i])
    
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
    
    def __init__(self, players: List[Take6Player], elo_system: EloSystem = None):
        self.players = players
        self.elo_system = elo_system or EloSystem()
        self.tournament_history = []
        self.round_number = 0
    
    def run_single_game(self, game_players: List[Take6Player], verbose: bool = False) -> Dict:
        """Run a single game with the given players."""
        game = Take6Game(num_players=len(game_players))
        state = game.reset()
        
        game_log = {
            'players': [p.player_id for p in game_players],
            'initial_elo': [p.elo_rating for p in game_players],
            'rounds': []
        }
        
        # Play 10 rounds (all cards)
        for round_num in range(10):
            player_actions = {}
            
            # Get action from each player
            for i, player in enumerate(game_players):
                valid_actions = game.get_valid_actions(i)
                card, row = player.get_action(state, valid_actions, training=True)
                player_actions[i] = (card, row)
            
            # Execute round
            round_results = game.play_round(player_actions)
            
            round_log = {
                'round': round_num,
                'actions': {i: (card.number, row) for i, (card, row) in player_actions.items()},
                'results': {i: (penalty, [c.number for c in cards]) 
                          for i, (penalty, cards) in round_results.items()},
                'penalty_totals': state.players_penalty_points.copy()
            }
            game_log['rounds'].append(round_log)
            
            if verbose:
                print(f"Round {round_num + 1}: Penalty totals: {state.players_penalty_points}")
        
        # Update Elo ratings
        self.elo_system.update_ratings(game_players, state.players_penalty_points)
        
        game_log['final_scores'] = state.players_penalty_points
        game_log['final_elo'] = [p.elo_rating for p in game_players]
        game_log['winner'] = state.get_winner()
        
        return game_log
    
    def run_round_robin(self, games_per_matchup: int = 1, verbose: bool = False) -> List[Dict]:
        """Run a round-robin tournament where every combination of 4 players plays."""
        results = []
        
        # Generate all combinations of 4 players
        from itertools import combinations
        matchups = list(combinations(self.players, 4))
        
        if verbose:
            print(f"Running round-robin with {len(matchups)} matchups, {games_per_matchup} games each")
        
        # Run games for each matchup
        for matchup in tqdm(matchups, desc="Tournament Progress"):
            for game_num in range(games_per_matchup):
                game_result = self.run_single_game(list(matchup), verbose)
                game_result['matchup_id'] = len(results)
                game_result['game_in_matchup'] = game_num
                results.append(game_result)
        
        self.tournament_history.extend(results)
        self.round_number += 1
        
        return results
    
    def run_random_games(self, num_games: int, players_per_game: int = 4, verbose: bool = False) -> List[Dict]:
        """Run random games with randomly selected players."""
        results = []
        
        if verbose:
            print(f"Running {num_games} random games with {players_per_game} players each")
        
        for game_num in tqdm(range(num_games), desc="Random Games"):
            # Randomly select players
            game_players = random.sample(self.players, players_per_game)
            
            game_result = self.run_single_game(game_players, verbose)
            game_result['game_id'] = game_num
            results.append(game_result)
        
        self.tournament_history.extend(results)
        
        return results
    
    def get_leaderboard(self) -> List[Tuple[int, float, int, float]]:
        """Get current leaderboard sorted by Elo rating."""
        leaderboard = []
        
        for player in self.players:
            leaderboard.append((
                player.player_id,
                player.elo_rating,
                player.games_played,
                player.get_average_score()
            ))
        
        # Sort by Elo rating (descending)
        leaderboard.sort(key=lambda x: x[1], reverse=True)
        
        return leaderboard
    
    def save_results(self, filename: str):
        """Save tournament results to file."""
        results = {
            'tournament_history': self.tournament_history,
            'final_leaderboard': self.get_leaderboard(),
            'round_number': self.round_number
        }
        
        with open(filename, 'w') as f:
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
    
    def __init__(self, players: List[Take6Player], elo_system: EloSystem = None, 
                 selection_ratio: float = 0.3, mutation_rate: float = 0.1):
        super().__init__(players, elo_system)
        self.selection_ratio = selection_ratio
        self.mutation_rate = mutation_rate
        self.generation = 0
    
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
        
        # Keep survivors
        for i, survivor in enumerate(survivors):
            # Reset some stats but keep Elo rating
            new_player = Take6Player(survivor.model, i, survivor.epsilon)
            new_player.elo_rating = survivor.elo_rating * 0.9  # Slight decay
            new_player.games_played = 0
            new_player.total_score = 0
            new_players.append(new_player)
        
        # Create offspring through mutation
        while len(new_players) < len(self.players):
            parent = random.choice(survivors)
            mutated_model = ModelFactory.mutate_model(parent.model, self.mutation_rate)
            
            new_player = Take6Player(mutated_model, len(new_players), parent.epsilon)
            new_player.elo_rating = parent.elo_rating * 0.8  # Offspring start lower
            new_players.append(new_player)
        
        self.players = new_players
        self.generation += 1
        
        return new_players
