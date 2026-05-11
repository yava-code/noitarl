"""
Custom Stable-Baselines3 callbacks for NoitaRL.

NoitaMonitorCallback  — episode stats, Telegram alerts, W&B logging, Rich table
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Optional

import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table
from stable_baselines3.common.callbacks import BaseCallback

from config import Config
from notify import TelegramNotifier

console = Console()

# optional wandb
try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False


class NoitaMonitorCallback(BaseCallback):
    """
    All-in-one callback:
      • logs episode stats to loguru + optional W&B
      • sends Telegram updates at configurable step intervals
      • prints a Rich summary table to the terminal
      • registers /status and /plot Telegram commands
    """

    def __init__(
        self,
        cfg: Config,
        notifier: TelegramNotifier,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._cfg       = cfg
        self._tg        = notifier
        self._start_ts  = time.time()
        self._last_tg   = 0          # steps at last telegram send
        self._ep_rewards:   deque[float] = deque(maxlen=500)
        self._ep_depths:    deque[float] = deque(maxlen=500)
        self._ep_lengths:   deque[int]   = deque(maxlen=500)
        self._ep_chunks:    deque[int]   = deque(maxlen=500)
        self._ep_distances: deque[float] = deque(maxlen=500)
        self._total_episodes = 0
        self._stop_requested = False

        # Register Telegram commands
        notifier.register_stats_provider(self._stats_text)
        notifier.register_command("stop",  self._cmd_stop)
        notifier.register_command("plot",  self._cmd_plot)
        notifier.register_command("help",  self._cmd_help)

    # ── SB3 lifecycle ─────────────────────────────────────────────────────────

    def _on_training_start(self) -> None:
        logger.info("Training started — {} steps, {} envs", self._cfg.total_timesteps, self._cfg.n_envs)
        self._tg.send_text(
            f"🚀 <b>NoitaRL training started</b>\n"
            f"Steps: {self._cfg.total_timesteps:,}\n"
            f"Envs: {self._cfg.n_envs}\n"
            f"LR: {self._cfg.learning_rate}\n"
            f"Commands: /status /plot /stop"
        )

    def _on_step(self) -> bool:
        if self._stop_requested:
            logger.warning("Stop requested via Telegram — ending training")
            self._tg.send_text("⛔ Training stopped on your request.")
            return False  # signals SB3 to stop

        # Harvest completed episode info from SB3's internal buffer
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is None:
                continue
            r = float(ep["r"])
            l = int(ep["l"])
            chunks = int(info.get("noita/visited_chunks", 0))
            dist   = float(info.get("noita/max_spawn_distance", 0.0))
            depth  = float(info.get("noita/max_depth", 0.0))
            self._ep_rewards.append(r)
            self._ep_lengths.append(l)
            self._ep_chunks.append(chunks)
            self._ep_distances.append(dist)
            self._total_episodes += 1

            # Log every episode to loguru
            logger.debug(
                "ep={} reward={:.2f} length={} chunks={} dist={:.0f} steps={}",
                self._total_episodes, r, l, chunks, dist, self.num_timesteps,
            )

            # Surface in SB3's "rollout/" namespace for TensorBoard
            self.logger.record("noita/visited_chunks",     chunks)
            self.logger.record("noita/max_spawn_distance", dist)
            self.logger.record("noita/max_depth",          depth)

            # W&B per-episode
            if self._cfg.wandb_enabled and _WANDB_OK:
                wandb.log({
                    "episode/reward":         r,
                    "episode/length":         l,
                    "episode/visited_chunks": chunks,
                    "episode/spawn_distance": dist,
                    "episode/max_depth":      depth,
                    "episode/total":          self._total_episodes,
                }, step=self.num_timesteps)

        # Periodic Telegram update
        steps_since = self.num_timesteps - self._last_tg
        if steps_since >= self._cfg.telegram_notify_every:
            self._last_tg = self.num_timesteps
            self._send_tg_update()

        return True

    def _on_rollout_end(self) -> None:
        if not self._ep_rewards:
            return

        mean_r = float(np.mean(self._ep_rewards))
        mean_l = float(np.mean(self._ep_lengths))
        mean_c = float(np.mean(self._ep_chunks))    if self._ep_chunks    else 0.0
        mean_d = float(np.mean(self._ep_distances)) if self._ep_distances else 0.0
        elapsed = time.time() - self._start_ts
        sps = self.num_timesteps / max(elapsed, 1)

        # W&B rollout metrics
        if self._cfg.wandb_enabled and _WANDB_OK:
            wandb.log({
                "rollout/mean_reward":          mean_r,
                "rollout/mean_ep_length":       mean_l,
                "rollout/mean_visited_chunks":  mean_c,
                "rollout/mean_spawn_distance":  mean_d,
                "rollout/steps_per_sec":        sps,
                "rollout/episodes":             self._total_episodes,
            }, step=self.num_timesteps)

        # Rich terminal table every N rollouts
        if self._total_episodes % 5 == 0:
            self._print_table(mean_r, mean_l, sps)

    def _on_training_end(self) -> None:
        elapsed = time.time() - self._start_ts
        mean_r = float(np.mean(self._ep_rewards)) if self._ep_rewards else 0.0
        msg = (
            f"✅ <b>Training complete!</b>\n"
            f"Steps: {self.num_timesteps:,}\n"
            f"Episodes: {self._total_episodes:,}\n"
            f"Mean reward (last {len(self._ep_rewards)}): {mean_r:.2f}\n"
            f"Time: {elapsed/3600:.1f} h"
        )
        logger.success("Training done. {}", msg.replace("\n", " | "))
        self._tg.send_text(msg)
        self._cmd_plot()  # send final plot

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _stats_text(self) -> str:
        elapsed = time.time() - self._start_ts
        pct = self.num_timesteps / max(self._cfg.total_timesteps, 1) * 100
        mean_r = float(np.mean(self._ep_rewards))   if self._ep_rewards   else 0.0
        mean_l = float(np.mean(self._ep_lengths))   if self._ep_lengths   else 0.0
        mean_c = float(np.mean(self._ep_chunks))    if self._ep_chunks    else 0.0
        mean_d = float(np.mean(self._ep_distances)) if self._ep_distances else 0.0
        sps = self.num_timesteps / max(elapsed, 1)
        return (
            f"Steps: {self.num_timesteps:,} / {self._cfg.total_timesteps:,} ({pct:.1f}%)\n"
            f"Episodes: {self._total_episodes:,}\n"
            f"Mean reward: {mean_r:.2f}\n"
            f"Mean ep length: {mean_l:.0f}\n"
            f"Mean visited chunks: {mean_c:.1f}\n"
            f"Mean spawn distance: {mean_d:.0f}\n"
            f"Speed: {sps:.0f} steps/s\n"
            f"Elapsed: {elapsed/3600:.1f} h"
        )

    def _send_tg_update(self) -> None:
        pct = self.num_timesteps / max(self._cfg.total_timesteps, 1) * 100
        mean_r = float(np.mean(self._ep_rewards)) if self._ep_rewards else 0.0
        elapsed = time.time() - self._start_ts
        eta_s = (self._cfg.total_timesteps - self.num_timesteps) / max(
            self.num_timesteps / max(elapsed, 1), 1
        )
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        self._tg.send_text(
            f"📈 <b>NoitaRL update</b>\n"
            f"[{bar}] {pct:.1f}%\n"
            f"Steps: {self.num_timesteps:,}\n"
            f"Mean reward: {mean_r:.2f}\n"
            f"ETA: {eta_s/3600:.1f} h"
        )

    def _print_table(self, mean_r: float, mean_l: float, sps: float) -> None:
        mean_c = float(np.mean(self._ep_chunks))    if self._ep_chunks    else 0.0
        mean_d = float(np.mean(self._ep_distances)) if self._ep_distances else 0.0
        t = Table(title=f"NoitaRL  step {self.num_timesteps:,}", style="dim")
        t.add_column("Metric", style="cyan")
        t.add_column("Value",  style="green")
        t.add_row("Episodes",            str(self._total_episodes))
        t.add_row("Mean reward",         f"{mean_r:.3f}")
        t.add_row("Mean ep length",      f"{mean_l:.0f}")
        t.add_row("Mean visited chunks", f"{mean_c:.1f}")
        t.add_row("Mean spawn distance", f"{mean_d:.0f}")
        t.add_row("Steps/sec",           f"{sps:.0f}")
        t.add_row("Elapsed",             f"{(time.time()-self._start_ts)/60:.1f} min")
        console.print(t)

    # ── Telegram commands ─────────────────────────────────────────────────────

    def _cmd_stop(self) -> None:
        logger.warning("Stop command received via Telegram")
        self._stop_requested = True
        self._tg.send_text("⛔ Stop signal received. Training will end after current rollout.")

    def _cmd_plot(self) -> None:
        if not self._ep_rewards:
            self._tg.send_text("No episode data yet.")
            return
        png = TelegramNotifier.make_reward_plot(
            list(self._ep_rewards),
            title=f"NoitaRL — {self.num_timesteps:,} steps",
        )
        self._tg.send_photo(png, caption=f"Episodes: {self._total_episodes}")

    def _cmd_help(self) -> None:
        self._tg.send_text(
            "🤖 <b>NoitaRL Bot commands</b>\n"
            "/status — current training stats\n"
            "/plot   — send reward graph\n"
            "/stop   — stop training gracefully\n"
            "/help   — this message"
        )
