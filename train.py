"""
Single-env training entry point.
For multi-env see train_multi.py.

Usage:
    python train.py
    python train.py --resume checkpoints/noita_ppo_200000_steps.zip
    python train.py --name "experiment-01"
"""

from __future__ import annotations

import argparse
import os
import sys

# Workaround for OpenMP duplicate library error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time

from loguru import logger
from rich.console import Console
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from callbacks import NoitaMonitorCallback
from config import Config
from noita_env import NoitaEnv
from notify import TelegramNotifier

console = Console()


def setup_logging(cfg: Config, run_name: str) -> None:
    os.makedirs(cfg.log_dir, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level=cfg.log_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    logger.add(
        os.path.join(cfg.log_dir, f"{run_name}.log"),
        level="DEBUG",
        rotation="100 MB",
        compression="zip",
        encoding="utf-8",
    )
    logger.info("Logging initialised → {}/{}.log", cfg.log_dir, run_name)


def setup_wandb(cfg: Config, run_name: str) -> None:
    if not cfg.wandb_enabled:
        return
    try:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            name=run_name,
            config={
                "total_timesteps": cfg.total_timesteps,
                "n_envs":          cfg.n_envs,
                "learning_rate":   cfg.learning_rate,
                "n_steps":         cfg.n_steps,
                "batch_size":      cfg.batch_size,
                "gamma":           cfg.gamma,
            },
            sync_tensorboard=True,
        )
        logger.info("W&B run started: {}/{}", cfg.wandb_project, run_name)
    except Exception as exc:
        logger.warning("W&B init failed (continuing without it): {}", exc)


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Return the most recently modified .zip in checkpoint_dir, or None."""
    import glob
    zips = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    return max(zips, key=os.path.getmtime) if zips else None


def train(args: argparse.Namespace) -> None:
    cfg = Config()
    if args.resume:
        cfg.resume_from = args.resume
    elif not args.fresh:
        latest = find_latest_checkpoint(cfg.checkpoint_dir)
        if latest:
            logger.info("Auto-resuming from latest checkpoint: {}", latest)
            cfg.resume_from = latest
        else:
            logger.info("No checkpoint found — starting fresh")
    else:
        logger.info("--fresh flag set — starting new run from scratch")
    if args.name:
        cfg.run_name = args.name

    run_name = cfg.effective_run_name()
    setup_logging(cfg, run_name)
    setup_wandb(cfg, run_name)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.tensorboard_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # ── Notifier ──────────────────────────────────────────────────────────────
    notifier = TelegramNotifier(cfg.telegram_token, cfg.telegram_chat_id)
    notifier.start_polling()

    # ── Environment ───────────────────────────────────────────────────────────
    logger.info("Creating NoitaEnv on port {}", cfg.noita_base_port)
    env = NoitaEnv(host=cfg.noita_host, port=cfg.noita_base_port)

    # ── Model ─────────────────────────────────────────────────────────────────
    if cfg.resume_from:
        logger.info("Resuming from {}", cfg.resume_from)
        try:
            model = PPO.load(cfg.resume_from, env=env, tensorboard_log=cfg.tensorboard_dir)
        except ValueError as exc:
            if "Observation spaces do not match" in str(exc):
                logger.warning(
                    "Checkpoint obs-space mismatch ({}) — obs space likely changed. "
                    "Starting fresh. Use --resume to force a specific checkpoint.",
                    exc,
                )
                cfg.resume_from = None
            else:
                raise
    if not cfg.resume_from:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "auto"
        logger.info("PPO device: {}", device)
        model = PPO(
            "MlpPolicy",
            env,
            verbose          = 1,
            learning_rate    = cfg.learning_rate,
            n_steps          = cfg.n_steps,
            batch_size       = cfg.batch_size,
            n_epochs         = cfg.n_epochs,
            gamma            = cfg.gamma,
            gae_lambda       = cfg.gae_lambda,
            clip_range       = cfg.clip_range,
            ent_coef         = cfg.ent_coef,
            vf_coef          = cfg.vf_coef,
            max_grad_norm    = cfg.max_grad_norm,
            tensorboard_log  = cfg.tensorboard_dir,
            device           = device,
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    monitor_cb = NoitaMonitorCallback(cfg, notifier, verbose=0)

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(cfg.checkpoint_freq // 1, 1),
        save_path   = cfg.checkpoint_dir,
        name_prefix = f"noita_ppo_{run_name}",
        verbose     = 1,
    )

    callbacks = CallbackList([monitor_cb, checkpoint_cb])

    # ── Train ─────────────────────────────────────────────────────────────────
    console.rule(f"[bold green]NoitaRL — {run_name}")
    console.print(f"  Steps:   [cyan]{cfg.total_timesteps:,}[/]")
    console.print(f"  Port:    [cyan]{cfg.noita_base_port}[/]")
    console.print(f"  W&B:     [cyan]{cfg.wandb_enabled}[/]")
    console.print(f"  TG:      [cyan]{cfg.telegram_enabled}[/]")
    console.print(f"  TBoard:  tensorboard --logdir {cfg.tensorboard_dir}")
    console.rule()

    try:
        model.learn(
            total_timesteps     = cfg.total_timesteps,
            callback            = callbacks,
            progress_bar        = True,
            reset_num_timesteps = cfg.resume_from is None,
            tb_log_name         = run_name,
        )
        logger.success("Training complete!")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    except Exception as exc:
        logger.exception("Training crashed: {}", exc)
        notifier.send_text(f"💥 <b>Training crashed!</b>\n{exc}")
        raise
    finally:
        out = os.path.join(cfg.checkpoint_dir, f"{run_name}_final")
        model.save(out)
        logger.info("Model saved → {}.zip", out)
        notifier.stop()

        if cfg.wandb_enabled:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NoitaRL — single-env PPO training")
    p.add_argument("--resume", type=str, default=None, metavar="PATH",
                   help="Resume from a specific .zip checkpoint")
    p.add_argument("--fresh",  action="store_true",
                   help="Ignore existing checkpoints and start a new run from scratch")
    p.add_argument("--name",   type=str, default=None, metavar="NAME",
                   help="Human-readable run name (also used in W&B / log file)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
