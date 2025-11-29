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

## Blender Battery Animation

This project also includes a Blender Python script for creating a battery charging animation.

### Features
- Animated battery charging visualization from 5% to 100%
- 10 color-coded bars (red → yellow → green gradient)
- Cool blinking/pulsing effects during charging
- Dynamic percentage text display
- Professional lighting and camera setup
- Bloom/glow post-processing effects

### Usage
1. Open Blender (2.80 or higher)
2. Go to the Scripting workspace
3. Open `blender_battery_animation.py`
4. Click "Run Script"

Or run from command line:
```bash
blender --python blender_battery_animation.py
```

### Rendering
- Press `Ctrl+F12` to render the full animation
- Output: 1920x1080, 30fps, H.264 MP4
- Duration: ~8 seconds (250 frames)

### Customization
Edit these parameters in the `main()` function:
- `num_bars`: Number of battery segments (default: 10)
- `start_percent`: Initial charge level (default: 5%)
- `end_percent`: Final charge level (default: 100%)
- `total_frames`: Animation duration (default: 250)
- `blink_speed`: Blinking frequency (default: 4)
