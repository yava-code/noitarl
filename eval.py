"""
Run a saved model against Noita in deterministic mode (no exploration).
Good for recording YouTube footage of the trained agent.

Usage:
    python eval.py noita_ppo_final.zip
    python eval.py checkpoints/noita_ppo_200000_steps.zip --episodes 20
    python eval.py model.zip --slow 0.5   # run at 50% speed (sleep 50ms between steps)
"""


import argparse
import sys
import time

from loguru import logger
from rich.console import Console
from rich.table import Table
from stable_baselines3 import PPO

from noita_env import NoitaEnv
from config import Config

console = Console()


import torch

def calculate_thinking(model, obs):
    """Return (wand_head_probs, saliency_per_obs_dim) for live HUD overlay.

    Mirrors ThinkingCallback._calculate_thinking — see that docstring for
    why we pick the wand head for visualisation under MultiDiscrete.
    """
    obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.device).requires_grad_(True)

    distribution = model.policy.get_distribution(obs_tensor)
    inner = distribution.distribution
    if isinstance(inner, (list, tuple)):
        wand_dist = inner[-1]
        probs  = wand_dist.probs[0].detach().cpu().numpy().tolist()
        logits = wand_dist.logits
    else:
        probs  = inner.probs[0].detach().cpu().numpy().tolist()
        logits = inner.logits

    action_idx = logits.argmax()
    logits[0, action_idx].backward()

    saliency = obs_tensor.grad.abs().squeeze().cpu().numpy().tolist()

    return probs, saliency

def evaluate(model_path: str, n_episodes: int, port: int, step_delay: float) -> None:
    cfg = Config()

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

    console.rule(f"[bold cyan]NoitaRL Evaluation — {model_path}")
    console.print(f"  Episodes:   [cyan]{n_episodes}[/]")
    console.print(f"  Port:       [cyan]{port}[/]")
    console.print(f"  Step delay: [cyan]{step_delay*1000:.0f} ms[/]")
    console.rule()

    env = NoitaEnv(host=cfg.noita_host, port=port)

    logger.info("Loading model from {}", model_path)
    model = PPO.load(model_path, env=env)

    ep_rewards = []
    ep_lengths = []
    ep_depths  = []
    
    best_ep_reward = -float('inf')
    best_ep_stats = {}

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done   = False
        total_r = 0.0
        steps   = 0

        while not done:
            # Calculate "Thinking" data
            probs, saliency = calculate_thinking(model, obs)
            env.set_extra({"probs": probs, "saliency": saliency})

            action, _ = model.predict(obs, deterministic=True)
            # MultiDiscrete: pass the action through as-is; env.step coerces to tuple.
            obs, r, done, _, _ = env.step(action)
            total_r += r
            steps   += 1
            if step_delay > 0:
                time.sleep(step_delay)

        ep_rewards.append(total_r)
        ep_lengths.append(steps)
        ep_depths.append(env.max_depth_y)
        logger.info("Ep {:3d}: reward={:8.2f}  steps={:5d}  max_depth={:.0f}",
                    ep, total_r, steps, env.max_depth_y)
                    
        if total_r > best_ep_reward:
            best_ep_reward = total_r
            best_ep_stats = {
                "ep": ep,
                "reward": total_r,
                "steps": steps,
                "max_depth": env.max_depth_y,
                "max_x": env.max_x,
                "kills": env.last_kills,
                "damage_taken": env.total_damage
            }

    # Summary table
    import numpy as np
    t = Table(title=f"Evaluation summary — {n_episodes} episodes", style="dim")
    t.add_column("Metric",    style="cyan")
    t.add_column("Mean",      style="green")
    t.add_column("Std",       style="yellow")
    t.add_column("Min",       style="red")
    t.add_column("Max",       style="green")
    for label, vals in [("Reward", ep_rewards), ("Length", ep_lengths), ("Max depth", ep_depths)]:
        a = np.array(vals, dtype=float)
        t.add_row(label, f"{a.mean():.2f}", f"{a.std():.2f}", f"{a.min():.2f}", f"{a.max():.2f}")
    console.print(t)

    if best_ep_stats:
        console.rule("[bold magenta]Best Episode Record")
        console.print(f"  Episode: [magenta]{best_ep_stats['ep']}[/]")
        console.print(f"  Reward:  [magenta]{best_ep_stats['reward']:.2f}[/]")
        console.print(f"  Steps:   [magenta]{best_ep_stats['steps']}[/]")
        console.print(f"  Depth:   [magenta]{best_ep_stats['max_depth']:.0f}[/]")
        console.print(f"  Max X:   [magenta]{best_ep_stats['max_x']:.0f}[/]")
        console.print(f"  Kills:   [magenta]{best_ep_stats['kills']}[/]")
        console.print(f"  Damage:  [magenta]{best_ep_stats['damage_taken']:.0f}[/]")
        console.rule()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NoitaRL evaluation / YouTube recording mode")
    p.add_argument("model",      type=str,               help="Path to .zip model")
    p.add_argument("--episodes", type=int, default=10,   help="Number of episodes to run")
    p.add_argument("--port",     type=int, default=5001, help="Noita WebSocket port")
    p.add_argument("--slow",     type=float, default=0.0,
                   help="Extra delay per step in seconds (e.g. 0.1 for slow-mo)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model, args.episodes, args.port, args.slow)
