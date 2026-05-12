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
import sys
import argparse
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from callbacks import NoitaMonitorCallback
from config import Config
from noita_env import NoitaEnv
from notify import TelegramNotifier
from video_recorder import VideoRecorder


def make_env(port: int):
    """Factory returning a callable that constructs NoitaEnv on the given port."""
    def _init():
        return NoitaEnv(port=port)
    return _init


def train(n_envs: int, base_port: int, total_steps: int, resume: str | None):
    cfg = Config()
    cfg.n_envs          = n_envs
    cfg.noita_base_port = base_port
    cfg.total_timesteps = total_steps
    if resume:
        cfg.resume_from = resume

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    run_name = cfg.effective_run_name()

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
    os.makedirs(cfg.log_dir, exist_ok=True)
    logger.add(f"{cfg.log_dir}/{run_name}.log", rotation="100 MB")

    notifier = TelegramNotifier(cfg.telegram_token, cfg.telegram_chat_id)
    notifier.start_polling()

    recorder = VideoRecorder(notifier, groq_api_key=cfg.groq_api_key)
    recorder.start()

    ports = [base_port + i for i in range(n_envs)]
    logger.info("Starting {} envs on ports {}", n_envs, ports)

    env = SubprocVecEnv([make_env(p) for p in ports])
    # Note: SubprocVecEnv forks processes so set_recorder can't propagate.
    # Per-step triggers come from callbacks instead (episode-level).

    monitor_cb    = NoitaMonitorCallback(cfg, notifier, recorder=recorder)
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(cfg.checkpoint_freq // n_envs, 1),
        save_path   = cfg.checkpoint_dir,
        name_prefix = f"noita_ppo_{n_envs}env",
        verbose     = 1,
    )
    callbacks = CallbackList([monitor_cb, checkpoint_cb])

    if resume:
        logger.info("Resuming from {}", resume)
        model = PPO.load(resume, env=env, tensorboard_log=cfg.tensorboard_dir)
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "auto"
        logger.info("PPO device: {}", device)
        model = PPO(
            "MlpPolicy", env,
            verbose=1, learning_rate=cfg.learning_rate, n_steps=cfg.n_steps,
            batch_size=cfg.batch_size, n_epochs=cfg.n_epochs, gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            tensorboard_log=cfg.tensorboard_dir,
            device=device,
        )

    logger.info("Training {}×{:,} steps, ~{:.1f}h per env",
                n_envs, total_steps, total_steps / n_envs / 18 / 3600)
    logger.info("TensorBoard: tensorboard --logdir {}", cfg.tensorboard_dir)

    try:
        model.learn(total_timesteps=total_steps, callback=callbacks,
                    progress_bar=True, reset_num_timesteps=resume is None,
                    tb_log_name=run_name)
        logger.success("Training complete!")
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
    except Exception as exc:
        logger.exception("Crashed: {}", exc)
        notifier.send_text(f"💥 Multi-train crashed: {exc}")
        raise
    finally:
        recorder.stop()
        out = os.path.join(cfg.checkpoint_dir, f"{run_name}_final")
        model.save(out)
        logger.info("Saved → {}.zip", out)
        notifier.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs",   type=int, default=2,         help="number of parallel Noita instances")
    ap.add_argument("--port",   type=int, default=5001,       help="base port (5001, 5002, …)")
    ap.add_argument("--steps",  type=int, default=1_000_000,  help="total training steps")
    ap.add_argument("--resume", type=str, default=None,       help="path to .zip checkpoint to resume")
    args = ap.parse_args()
    train(args.envs, args.port, args.steps, args.resume)
