# Take 6 Tournament - Machine Learning Project

This project implements a tournament system for the card game "Take 6" (6 nimmt!) using TensorFlow neural networks. 40 model instances compete against each other in a tournament format using the Elo rating system for reinforcement learning.

## Game Rules
Take 6 is a card game where players try to avoid taking penalty points by playing cards strategically onto four rows. Each row can hold up to 5 cards, and when a 6th card is played, the player must take all cards in that row.

## Project Structure
- `game/` - Core game logic and rules
- `models/` - Neural network implementations
- `tournament/` - Tournament system and Elo rating
- `training/` - Training scripts and utilities
- `analysis/` - Performance analysis and visualization

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python main.py`
3. View results: `python analysis/visualize_results.py`

## Features
- Complete Take 6 game implementation
- TensorFlow-based neural network players
- Tournament system with 40 competing models
- Elo rating system for continuous improvement
- Performance tracking and visualization
