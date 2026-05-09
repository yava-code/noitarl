"""
Multi-environment parallel training.

Each environment = one Noita instance on its own port.
Noita reads the port from mods/noitarl/port.txt.

Setup for N instances:
  1. Create N copies of Noita (or N Steam library folders, or use -applaunch with
     different -gamedir flags if available).
  2. In each copy's mods/noitarl/port.txt write a unique port: 5001, 5002, ...
  3. Start all Noita instances with the noitarl mod enabled.
  4. Run this script.

Single-machine quick test (2 instances):
  - Copy of Noita A: port.txt = 5001
  - Copy of Noita B: port.txt = 5002
  - python train_multi.py --envs 2
"""

import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from noita_env import NoitaEnv


def make_env(port: int):
    """Factory returning a callable that constructs NoitaEnv on the given port."""
    def _init():
        return NoitaEnv(port=port)
    return _init


def train(n_envs: int, base_port: int, total_steps: int, resume: str | None):
    os.makedirs("checkpoints", exist_ok=True)

    ports = [base_port + i for i in range(n_envs)]
    print(f"[MultiTrain] Starting {n_envs} envs on ports {ports}")

    env = SubprocVecEnv([make_env(p) for p in ports])

    checkpoint_cb = CheckpointCallback(
        save_freq   = 100_000 // n_envs,   # wall-clock steps per env
        save_path   = "./checkpoints/",
        name_prefix = f"noita_ppo_{n_envs}env",
        verbose     = 1,
    )

    if resume:
        print(f"[MultiTrain] Resuming from {resume}")
        model = PPO.load(resume, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose         = 1,
            learning_rate   = 3e-4,
            n_steps         = 2048,
            batch_size      = 64,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            tensorboard_log = "./noita_ppo_tensorboard/",
        )

    print(f"[MultiTrain] Training for {total_steps:,} steps "
          f"(≈ {total_steps / n_envs / 18 / 3600:.1f} h per env at 18 it/s)")
    print("[MultiTrain] TensorBoard:  tensorboard --logdir ./noita_ppo_tensorboard/")

    try:
        model.learn(
            total_timesteps = total_steps,
            callback        = checkpoint_cb,
            progress_bar    = True,
            reset_num_timesteps = resume is None,
        )
        print("[MultiTrain] Done!")
    except KeyboardInterrupt:
        print("[MultiTrain] Interrupted.")

    out = f"noita_ppo_{n_envs}env_final"
    model.save(out)
    print(f"[MultiTrain] Saved → {out}.zip")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs",   type=int, default=2,         help="number of parallel Noita instances")
    ap.add_argument("--port",   type=int, default=5001,       help="base port (5001, 5002, …)")
    ap.add_argument("--steps",  type=int, default=1_000_000,  help="total training steps")
    ap.add_argument("--resume", type=str, default=None,       help="path to .zip checkpoint to resume")
    args = ap.parse_args()
    train(args.envs, args.port, args.steps, args.resume)
