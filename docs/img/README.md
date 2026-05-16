# Assets

This directory contains images for the README and documentation.

## Required Images

- `demo_placeholder.gif` — GIF of the RL agent playing Noita
- `training_curve_placeholder.png` — Plot of training reward over time

## How to Generate

### Demo GIF
1. Run training with recording enabled
2. Use OBS or similar to capture gameplay
3. Convert to GIF with ffmpeg:
   ```bash
   ffmpeg -i recording.mp4 -vf "fps=10,scale=480:-1:flags=lanczos" demo.gif
   ```

### Training Curve
```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/episode_history.csv')
plt.figure(figsize=(10, 6))
plt.plot(df['episode'], df['reward'], alpha=0.3, label='Raw')
plt.plot(df['episode'], df['reward'].rolling(100).mean(), label='Smoothed')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.legend()
plt.savefig('docs/img/training_curve.png', dpi=150)
```
