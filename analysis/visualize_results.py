"""
Analysis and visualization tools for tournament results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from typing import List, Dict
import os

class TournamentAnalyzer:
    """Analyzer for tournament results and player performance."""
    
    def __init__(self, results_file: str = None):
        self.results = None
        if results_file and os.path.exists(results_file):
            self.load_results(results_file)
    
    def load_results(self, filename: str):
        """Load tournament results from file."""
        with open(filename, 'r') as f:
            self.results = json.load(f)
    
    def create_performance_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with game-by-game performance data."""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        
        for game_idx, game in enumerate(self.results['tournament_history']):
            for player_idx, player_id in enumerate(game['players']):
                data.append({
                    'game_id': game_idx,
                    'player_id': player_id,
                    'initial_elo': game['initial_elo'][player_idx],
                    'final_elo': game['final_elo'][player_idx],
                    'elo_change': game['final_elo'][player_idx] - game['initial_elo'][player_idx],
                    'penalty_points': game['final_scores'][player_idx],
                    'won_game': game['winner'] == player_idx,
                    'rank': sorted(enumerate(game['final_scores']), key=lambda x: x[1])[player_idx][0] + 1
                })
        
        return pd.DataFrame(data)
    
    def plot_elo_evolution(self, player_ids: List[int] = None, save_path: str = None):
        """Plot Elo rating evolution over time."""
        df = self.create_performance_dataframe()
        
        if df.empty:
            print("No data to plot")
            return
        
        if player_ids is None:
            # Plot top 10 players by final Elo
            final_elos = df.groupby('player_id')['final_elo'].last().sort_values(ascending=False)
            player_ids = final_elos.head(10).index.tolist()
        
        plt.figure(figsize=(12, 8))
        
        for player_id in player_ids:
            player_data = df[df['player_id'] == player_id].sort_values('game_id')
            if not player_data.empty:
                plt.plot(player_data['game_id'], player_data['final_elo'], 
                        label=f'Player {player_id}', alpha=0.7, linewidth=2)
        
        plt.xlabel('Game Number')
        plt.ylabel('Elo Rating')
        plt.title('Elo Rating Evolution Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_score_distribution(self, save_path: str = None):
        """Plot distribution of penalty points across all games."""
        df = self.create_performance_dataframe()
        
        if df.empty:
            print("No data to plot")
            return
        
        plt.figure(figsize=(10, 6))
        
        sns.histplot(data=df, x='penalty_points', bins=30, kde=True)
        plt.xlabel('Penalty Points')
        plt.ylabel('Frequency')
        plt.title('Distribution of Penalty Points')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_score = df['penalty_points'].mean()
        median_score = df['penalty_points'].median()
        plt.axvline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.1f}')
        plt.axvline(median_score, color='orange', linestyle='--', label=f'Median: {median_score:.1f}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_elo_vs_performance(self, save_path: str = None):
        """Plot correlation between Elo rating and average performance."""
        df = self.create_performance_dataframe()
        
        if df.empty:
            print("No data to plot")
            return
        
        # Calculate average performance metrics per player
        player_stats = df.groupby('player_id').agg({
            'final_elo': 'last',
            'penalty_points': 'mean',
            'won_game': 'mean',
            'rank': 'mean'
        }).reset_index()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Elo vs Average Penalty Points
        axes[0, 0].scatter(player_stats['final_elo'], player_stats['penalty_points'], alpha=0.6)
        axes[0, 0].set_xlabel('Final Elo Rating')
        axes[0, 0].set_ylabel('Average Penalty Points')
        axes[0, 0].set_title('Elo vs Average Penalty Points')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Elo vs Win Rate
        axes[0, 1].scatter(player_stats['final_elo'], player_stats['won_game'], alpha=0.6)
        axes[0, 1].set_xlabel('Final Elo Rating')
        axes[0, 1].set_ylabel('Win Rate')
        axes[0, 1].set_title('Elo vs Win Rate')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Elo vs Average Rank
        axes[1, 0].scatter(player_stats['final_elo'], player_stats['rank'], alpha=0.6)
        axes[1, 0].set_xlabel('Final Elo Rating')
        axes[1, 0].set_ylabel('Average Rank')
        axes[1, 0].set_title('Elo vs Average Rank')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Elo distribution
        axes[1, 1].hist(player_stats['final_elo'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Final Elo Rating')
        axes[1, 1].set_ylabel('Number of Players')
        axes[1, 1].set_title('Final Elo Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self, save_path: str = None):
        """Plot learning curves showing improvement over time."""
        df = self.create_performance_dataframe()
        
        if df.empty:
            print("No data to plot")
            return
        
        # Calculate rolling averages
        window = 50
        df['rolling_penalty'] = df.groupby('player_id')['penalty_points'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df['rolling_elo'] = df.groupby('player_id')['final_elo'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot for top 5 players
        top_players = df.groupby('player_id')['final_elo'].last().nlargest(5).index
        
        for player_id in top_players:
            player_data = df[df['player_id'] == player_id].sort_values('game_id')
            
            axes[0].plot(player_data['game_id'], player_data['rolling_penalty'], 
                        label=f'Player {player_id}', alpha=0.7)
            axes[1].plot(player_data['game_id'], player_data['rolling_elo'], 
                        label=f'Player {player_id}', alpha=0.7)
        
        axes[0].set_xlabel('Game Number')
        axes[0].set_ylabel('Rolling Average Penalty Points')
        axes[0].set_title(f'Learning Curves - Penalty Points (Rolling Average, window={window})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Game Number')
        axes[1].set_ylabel('Rolling Average Elo')
        axes[1].set_title(f'Learning Curves - Elo Rating (Rolling Average, window={window})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, output_dir: str = 'analysis_output'):
        """Generate a comprehensive analysis report."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating tournament analysis report...")
        
        # Create plots
        self.plot_elo_evolution(save_path=os.path.join(output_dir, 'elo_evolution.png'))
        self.plot_score_distribution(save_path=os.path.join(output_dir, 'score_distribution.png'))
        self.plot_elo_vs_performance(save_path=os.path.join(output_dir, 'elo_vs_performance.png'))
        self.plot_learning_curves(save_path=os.path.join(output_dir, 'learning_curves.png'))
        
        # Generate summary statistics
        df = self.create_performance_dataframe()
        
        if not df.empty:
            summary = {
                'total_games': len(df['game_id'].unique()),
                'total_players': len(df['player_id'].unique()),
                'average_penalty_points': float(df['penalty_points'].mean()),
                'std_penalty_points': float(df['penalty_points'].std()),
                'elo_range': {
                    'min': float(df['final_elo'].min()),
                    'max': float(df['final_elo'].max()),
                    'mean': float(df['final_elo'].mean())
                },
                'top_10_players': df.groupby('player_id')['final_elo'].last().nlargest(10).to_dict()
            }
            
            with open(os.path.join(output_dir, 'summary_stats.json'), 'w') as f:
                json.dump(summary, f, indent=2)
        
        print(f"Analysis report saved to {output_dir}/")

def create_sample_visualization():
    """Create sample visualizations with dummy data for demonstration."""
    print("Creating sample visualizations...")
    
    # Generate dummy data
    np.random.seed(42)
    games = 1000
    players = 40
    
    data = []
    for game in range(games):
        for player in range(players):
            # Simulate some learning - better players improve over time
            base_skill = np.random.normal(0, 1)  # Player's inherent skill
            learning_factor = game / games * 0.5  # Learning over time
            
            penalty = max(1, int(np.random.gamma(2, 10) + base_skill * 5 - learning_factor * 10))
            elo = 1500 + base_skill * 200 + learning_factor * 100 + np.random.normal(0, 50)
            
            data.append({
                'game_id': game,
                'player_id': player,
                'penalty_points': penalty,
                'elo': elo
            })
    
    df = pd.DataFrame(data)
    
    # Create visualizations
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Elo evolution for top 5 players
    top_players = df.groupby('player_id')['elo'].last().nlargest(5).index
    for player in top_players:
        player_data = df[df['player_id'] == player]
        axes[0, 0].plot(player_data['game_id'], player_data['elo'], 
                       label=f'Player {player}', alpha=0.7)
    axes[0, 0].set_title('Elo Evolution - Top 5 Players')
    axes[0, 0].set_xlabel('Game Number')
    axes[0, 0].set_ylabel('Elo Rating')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Penalty points distribution
    axes[0, 1].hist(df['penalty_points'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Distribution of Penalty Points')
    axes[0, 1].set_xlabel('Penalty Points')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Elo vs average performance
    player_stats = df.groupby('player_id').agg({
        'elo': 'last',
        'penalty_points': 'mean'
    }).reset_index()
    axes[1, 0].scatter(player_stats['elo'], player_stats['penalty_points'], alpha=0.6)
    axes[1, 0].set_title('Final Elo vs Average Penalty Points')
    axes[1, 0].set_xlabel('Final Elo Rating')
    axes[1, 0].set_ylabel('Average Penalty Points')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final Elo distribution
    axes[1, 1].hist(player_stats['elo'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Final Elo Distribution')
    axes[1, 1].set_xlabel('Final Elo Rating')
    axes[1, 1].set_ylabel('Number of Players')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Take 6 Tournament Analysis - Sample Data', fontsize=16)
    plt.tight_layout()
    plt.savefig('sample_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Sample visualization saved as 'sample_analysis.png'")

if __name__ == "__main__":
    create_sample_visualization()
