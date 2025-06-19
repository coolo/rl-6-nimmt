"""
Checkpoint-based analysis and visualization tools for tournament results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
import glob
from typing import List, Dict, Optional, Tuple
from datetime import datetime

class CheckpointAnalyzer:
    """Analyzer for checkpoint data and ELO progression logs."""
    
    def __init__(self):
        self.elo_data = None
        self.checkpoint_data = {}
        self.session_history = {}
        
    def find_latest_session(self) -> Optional[str]:
        """Find the most recent training session."""
        # Look for ELO progression files
        elo_files = glob.glob("logs/*_elo_progression.csv")
        if not elo_files:
            return None
        
        # Get the most recent file
        latest_file = max(elo_files, key=os.path.getmtime)
        # Extract session name from filename
        session_name = os.path.basename(latest_file).replace("_elo_progression.csv", "")
        return session_name
    
    def load_session_data(self, session_name: str = None):
        """Load all data for a training session."""
        if session_name is None:
            session_name = self.find_latest_session()
            if session_name is None:
                raise ValueError("No training sessions found")
        
        print(f"Loading data for session: {session_name}")
        
        # Load ELO progression data
        elo_file = f"logs/{session_name}_elo_progression.csv"
        if os.path.exists(elo_file):
            self.elo_data = pd.read_csv(elo_file)
            print(f"✅ Loaded ELO progression data: {len(self.elo_data)} records")
        else:
            print(f"❌ ELO progression file not found: {elo_file}")
            self.elo_data = pd.DataFrame()
        
        # Load checkpoint data
        checkpoint_dir = f"checkpoints/{session_name}"
        if os.path.exists(checkpoint_dir):
            self._load_checkpoint_data(checkpoint_dir)
        else:
            print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        
        # Load session history
        history_file = f"{checkpoint_dir}/session_history.json"
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                self.session_history = json.load(f)
            print(f"✅ Loaded session history")
        
        return session_name
    
    def _load_checkpoint_data(self, checkpoint_dir: str):
        """Load data from all checkpoints in a session."""
        for item in os.listdir(checkpoint_dir):
            if item.startswith('cycle_') and os.path.isdir(os.path.join(checkpoint_dir, item)):
                cycle_dir = os.path.join(checkpoint_dir, item)
                cycle_num = int(item.split('_')[1])
                
                # Load checkpoint info
                info_file = os.path.join(cycle_dir, "checkpoint_info.json")
                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        self.checkpoint_data[cycle_num] = json.load(f)
        
        print(f"✅ Loaded {len(self.checkpoint_data)} checkpoints")
    
    def plot_elo_progression(self, top_n: int = 10, save_path: str = None):
        """Plot ELO progression over time for top players."""
        if self.elo_data.empty:
            print("No ELO data available for plotting")
            return
        
        # Find top players by final ELO
        final_elos = self.elo_data.groupby('player_id')['elo_rating'].last().sort_values(ascending=False)
        top_players = final_elos.head(top_n).index.tolist()
        
        plt.figure(figsize=(14, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_players)))
        
        for i, player_id in enumerate(top_players):
            player_data = self.elo_data[self.elo_data['player_id'] == player_id].sort_values('match_number')
            
            if not player_data.empty:
                plt.plot(player_data['match_number'], player_data['elo_rating'], 
                        label=f'Player {player_id} (Final: {final_elos[player_id]:.0f})', 
                        alpha=0.8, linewidth=2, color=colors[i])
        
        plt.xlabel('Match Number')
        plt.ylabel('ELO Rating')
        plt.title(f'ELO Rating Progression - Top {top_n} Players')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_elo_by_cycle(self, save_path: str = None):
        """Plot ELO progression by training cycle."""
        if self.elo_data.empty:
            print("No ELO data available for plotting")
            return
        
        # Group by cycle and get stats
        cycle_stats = self.elo_data.groupby('cycle')['elo_rating'].agg(['mean', 'std', 'max', 'min']).reset_index()
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(cycle_stats['cycle'], cycle_stats['mean'], 'o-', label='Average ELO', linewidth=2)
        plt.fill_between(cycle_stats['cycle'], 
                        cycle_stats['mean'] - cycle_stats['std'], 
                        cycle_stats['mean'] + cycle_stats['std'], 
                        alpha=0.3, label='±1 Std Dev')
        plt.plot(cycle_stats['cycle'], cycle_stats['max'], 's-', label='Best Player', alpha=0.7)
        plt.plot(cycle_stats['cycle'], cycle_stats['min'], '^-', label='Worst Player', alpha=0.7)
        
        plt.xlabel('Training Cycle')
        plt.ylabel('ELO Rating')
        plt.title('ELO Rating Distribution by Training Cycle')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_score_vs_elo(self, save_path: str = None):
        """Plot relationship between average score and ELO rating."""
        if self.elo_data.empty:
            print("No ELO data available for plotting")
            return
        
        # Get final data for each player
        final_data = self.elo_data.groupby('player_id').last()
        
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(final_data['avg_score'], final_data['elo_rating'], 
                            c=final_data['games_played'], cmap='viridis', 
                            alpha=0.7, s=60)
        
        plt.colorbar(scatter, label='Games Played')
        plt.xlabel('Average Penalty Score (lower is better)')
        plt.ylabel('ELO Rating')
        plt.title('ELO Rating vs Average Score')
        plt.grid(True, alpha=0.3)
        
        # Add correlation info
        correlation = final_data['avg_score'].corr(final_data['elo_rating'])
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_checkpoint_evolution(self, save_path: str = None):
        """Plot evolution of best players across checkpoints."""
        if not self.checkpoint_data:
            print("No checkpoint data available for plotting")
            return
        
        cycles = sorted(self.checkpoint_data.keys())
        best_elos = []
        avg_elos = []
        elo_spreads = []
        
        for cycle in cycles:
            stats = self.checkpoint_data[cycle]['statistics']
            best_elos.append(stats['max_elo'])
            avg_elos.append(stats['avg_elo'])
            elo_spreads.append(stats['elo_spread'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot ELO evolution
        ax1.plot(cycles, best_elos, 'o-', label='Best Player', linewidth=2)
        ax1.plot(cycles, avg_elos, 's-', label='Average ELO', linewidth=2)
        ax1.set_xlabel('Checkpoint Cycle')
        ax1.set_ylabel('ELO Rating')
        ax1.set_title('ELO Evolution Across Checkpoints')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot ELO spread
        ax2.plot(cycles, elo_spreads, '^-', color='orange', linewidth=2)
        ax2.set_xlabel('Checkpoint Cycle')
        ax2.set_ylabel('ELO Spread (Max - Min)')
        ax2.set_title('ELO Diversity Across Checkpoints')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_elo_distribution(self, save_path: str = None):
        """Plot ELO distribution across all players."""
        if self.elo_data.empty:
            print("No ELO data available for plotting")
            return
        
        # Get final ELO ratings for each player
        final_elos = self.elo_data.groupby('player_id')['elo_rating'].last()
        
        plt.figure(figsize=(10, 6))
        
        plt.hist(final_elos, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(final_elos.mean(), color='red', linestyle='--', 
                   label=f'Mean: {final_elos.mean():.1f}')
        plt.axvline(final_elos.median(), color='orange', linestyle='--',
                   label=f'Median: {final_elos.median():.1f}')
        
        plt.xlabel('Final ELO Rating')
        plt.ylabel('Number of Players')
        plt.title('Distribution of Final ELO Ratings')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_win_rates(self, save_path: str = None):
        """Plot win rates by ELO rating."""
        if self.elo_data.empty:
            print("No ELO data available for plotting")
            return
        
        # Calculate win rates (assuming we have game outcome data)
        # For now, we'll use ELO change as a proxy for performance
        win_proxy = self.elo_data.groupby('player_id').agg({
            'elo_rating': ['last', 'count'],
            'elo_change': 'mean'
        }).round(2)
        
        win_proxy.columns = ['final_elo', 'games_played', 'avg_elo_change']
        win_proxy = win_proxy.reset_index()
        
        plt.figure(figsize=(10, 6))
        
        scatter = plt.scatter(win_proxy['final_elo'], win_proxy['avg_elo_change'], 
                             s=win_proxy['games_played']*2, alpha=0.6)
        
        plt.xlabel('Final ELO Rating')
        plt.ylabel('Average ELO Change per Game')
        plt.title('Performance vs Rating (Bubble size = Games Played)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_metrics(self, save_path: str = None):
        """Plot various performance metrics over time."""
        if self.elo_data.empty:
            print("No ELO data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. ELO spread over time
        match_stats = self.elo_data.groupby('match_number')['elo_rating'].agg(['std', 'max', 'min'])
        match_stats['spread'] = match_stats['max'] - match_stats['min']
        
        axes[0, 0].plot(match_stats.index, match_stats['spread'])
        axes[0, 0].set_title('ELO Spread Over Time')
        axes[0, 0].set_xlabel('Match Number')
        axes[0, 0].set_ylabel('ELO Range (Max - Min)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Standard deviation over time
        axes[0, 1].plot(match_stats.index, match_stats['std'])
        axes[0, 1].set_title('ELO Standard Deviation Over Time')
        axes[0, 1].set_xlabel('Match Number')
        axes[0, 1].set_ylabel('ELO Standard Deviation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Games played distribution
        games_per_player = self.elo_data['player_id'].value_counts()
        axes[1, 0].hist(games_per_player, bins=15, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Games Played Distribution')
        axes[1, 0].set_xlabel('Games Played per Player')
        axes[1, 0].set_ylabel('Number of Players')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. ELO improvement histogram
        elo_improvements = self.elo_data.groupby('player_id')['elo_rating'].agg(lambda x: x.iloc[-1] - x.iloc[0])
        axes[1, 1].hist(elo_improvements, bins=15, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(elo_improvements.mean(), color='red', linestyle='--',
                          label=f'Mean: {elo_improvements.mean():.1f}')
        axes[1, 1].set_title('ELO Improvement Distribution')
        axes[1, 1].set_xlabel('Total ELO Change')
        axes[1, 1].set_ylabel('Number of Players')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self) -> Dict:
        """Generate a comprehensive summary report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {},
            'performance_metrics': {},
            'training_progress': {}
        }
        
        # Data summary
        if not self.elo_data.empty:
            report['data_summary'] = {
                'total_matches': self.elo_data['match_number'].max(),
                'total_cycles': self.elo_data['cycle'].max(),
                'unique_players': self.elo_data['player_id'].nunique(),
                'total_elo_records': len(self.elo_data)
            }
            
            # Performance metrics
            final_data = self.elo_data.groupby('player_id').last()
            report['performance_metrics'] = {
                'best_player': {
                    'id': int(final_data['elo_rating'].idxmax()),
                    'elo': float(final_data['elo_rating'].max()),
                    'avg_score': float(final_data.loc[final_data['elo_rating'].idxmax(), 'avg_score'])
                },
                'worst_player': {
                    'id': int(final_data['elo_rating'].idxmin()),
                    'elo': float(final_data['elo_rating'].min()),
                    'avg_score': float(final_data.loc[final_data['elo_rating'].idxmin(), 'avg_score'])
                },
                'population_stats': {
                    'avg_elo': float(final_data['elo_rating'].mean()),
                    'std_elo': float(final_data['elo_rating'].std()),
                    'elo_range': float(final_data['elo_rating'].max() - final_data['elo_rating'].min())
                }
            }
        
        # Training progress
        if self.checkpoint_data:
            cycles = sorted(self.checkpoint_data.keys())
            if cycles:
                first_cycle = self.checkpoint_data[cycles[0]]
                last_cycle = self.checkpoint_data[cycles[-1]]
                
                report['training_progress'] = {
                    'checkpoints_created': len(cycles),
                    'first_checkpoint': cycles[0],
                    'last_checkpoint': cycles[-1],
                    'elo_improvement': {
                        'best_player': last_cycle['statistics']['max_elo'] - first_cycle['statistics']['max_elo'],
                        'average': last_cycle['statistics']['avg_elo'] - first_cycle['statistics']['avg_elo']
                    }
                }
        
        return report
    
    def generate_report(self, output_dir: str = "analysis_output"):
        """Generate a complete analysis report with visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📊 Generating analysis report in {output_dir}/")
        
        # Generate plots
        try:
            print("  📈 Creating ELO progression plot...")
            self.plot_elo_progression(save_path=f"{output_dir}/elo_progression.png")
            
            print("  📊 Creating cycle-based ELO plot...")
            self.plot_elo_by_cycle(save_path=f"{output_dir}/elo_by_cycle.png")
            
            print("  🎯 Creating score vs ELO plot...")
            self.plot_score_vs_elo(save_path=f"{output_dir}/score_vs_elo.png")
            
            if self.checkpoint_data:
                print("  💾 Creating checkpoint evolution plot...")
                self.plot_checkpoint_evolution(save_path=f"{output_dir}/checkpoint_evolution.png")
            
        except Exception as e:
            print(f"⚠️  Error creating plots: {e}")
        
        # Generate summary report
        try:
            print("  📋 Generating summary report...")
            report = self.generate_summary_report()
            
            with open(f"{output_dir}/summary_report.json", 'w') as f:
                json.dump(report, f, indent=2)
            
            # Create a human-readable summary
            self._create_text_summary(report, f"{output_dir}/summary.txt")
            
        except Exception as e:
            print(f"⚠️  Error generating summary: {e}")
        
        print(f"✅ Analysis complete! Check {output_dir}/ for results.")
    
    def _create_text_summary(self, report: Dict, filename: str):
        """Create a human-readable text summary."""
        with open(filename, 'w') as f:
            f.write("Take 6 RL Training Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            
            # Data summary
            if 'data_summary' in report and report['data_summary']:
                data = report['data_summary']
                f.write(f"Data Summary:\n")
                f.write(f"  Total Matches: {data.get('total_matches', 'N/A')}\n")
                f.write(f"  Training Cycles: {data.get('total_cycles', 'N/A')}\n")
                f.write(f"  Unique Players: {data.get('unique_players', 'N/A')}\n")
                f.write(f"  ELO Records: {data.get('total_elo_records', 'N/A')}\n\n")
            
            # Performance metrics
            if 'performance_metrics' in report and report['performance_metrics']:
                perf = report['performance_metrics']
                f.write(f"Performance Metrics:\n")
                
                if 'best_player' in perf:
                    best = perf['best_player']
                    f.write(f"  Best Player: {best['id']} (ELO: {best['elo']:.1f}, Avg Score: {best['avg_score']:.1f})\n")
                
                if 'worst_player' in perf:
                    worst = perf['worst_player']
                    f.write(f"  Worst Player: {worst['id']} (ELO: {worst['elo']:.1f}, Avg Score: {worst['avg_score']:.1f})\n")
                
                if 'population_stats' in perf:
                    pop = perf['population_stats']
                    f.write(f"  Population Average ELO: {pop['avg_elo']:.1f}\n")
                    f.write(f"  ELO Standard Deviation: {pop['std_elo']:.1f}\n")
                    f.write(f"  ELO Range: {pop['elo_range']:.1f}\n\n")
            
            # Training progress
            if 'training_progress' in report and report['training_progress']:
                progress = report['training_progress']
                f.write(f"Training Progress:\n")
                f.write(f"  Checkpoints Created: {progress.get('checkpoints_created', 'N/A')}\n")
                
                if 'elo_improvement' in progress:
                    imp = progress['elo_improvement']
                    f.write(f"  Best Player ELO Improvement: {imp['best_player']:.1f}\n")
                    f.write(f"  Average ELO Improvement: {imp['average']:.1f}\n")
            
            f.write(f"\nGenerated: {report['timestamp']}\n")
    
    def generate_comprehensive_report(self, output_dir: str = "analysis_output"):
        """Generate a comprehensive analysis report with all visualizations and insights."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📊 Generating comprehensive report in {output_dir}/")
        
        # Generate all plots
        if not self.elo_data.empty:
            self.plot_elo_progression(top_n=10, save_path=f"{output_dir}/elo_progression.png")
            self.plot_elo_distribution(save_path=f"{output_dir}/elo_distribution.png")
            self.plot_win_rates(save_path=f"{output_dir}/win_rates.png")
            self.plot_performance_metrics(save_path=f"{output_dir}/performance_metrics.png")
        
      
        # Generate summary report
        report = self.generate_summary_report()
        
        # Save summary to text file
        summary_file = f"{output_dir}/summary_report.txt"
        #self._save_summary_report(report, summary_file)
        
        # Save raw data as JSON
        raw_data_file = f"{output_dir}/raw_analysis_data.json"
        self._save_raw_data(report, raw_data_file)
        
        print(f"✅ Comprehensive report generated:")
        print(f"   📈 Visualizations: {output_dir}/*.png")
        print(f"   📋 Summary: {summary_file}")
        print(f"   📊 Raw Data: {raw_data_file}")
        
        return report
    
    def _save_raw_data(self, report: Dict, filename: str):
        """Save raw analysis data as JSON for further processing."""
        raw_data = {
            'session_info': report.get('session_info', {}),
            'player_statistics': report.get('player_statistics', {}),
            'training_progress': report.get('training_progress', {}),
            'timestamp': report.get('timestamp', ''),
            'elo_data_summary': {
                'total_matches': len(self.elo_data) if not self.elo_data.empty else 0,
                'unique_players': self.elo_data['player_id'].nunique() if not self.elo_data.empty else 0,
                'elo_range': {
                    'min': float(self.elo_data['elo_rating'].min()) if not self.elo_data.empty else 0,
                    'max': float(self.elo_data['elo_rating'].max()) if not self.elo_data.empty else 0
                }
            } if not self.elo_data.empty else {},
            'checkpoint_summary': {
                'total_checkpoints': len(self.checkpoint_data),
                'checkpoint_cycles': list(self.checkpoint_data.keys())
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(raw_data, f, indent=2, default=str)
