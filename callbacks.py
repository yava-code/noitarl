"""
Custom Stable-Baselines3 callbacks for NoitaRL.

NoitaMonitorCallback  — episode stats, Telegram alerts, W&B logging, Rich table
"""

from __future__ import annotations

import os
import time
import csv
from collections import deque
from typing import Optional

import glob
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
        recorder=None,
    ):
        super().__init__(verbose)
        self._cfg       = cfg
        self._tg        = notifier
        self._recorder  = recorder    # VideoRecorder | None
        self._start_ts  = time.time()
        self._last_tg   = 0          # steps at last telegram send
        self._muted     = False      # toggle periodic updates
        self._ep_rewards:   deque[float] = deque(maxlen=500)
        self._ep_depths:    deque[float] = deque(maxlen=500)
        self._ep_lengths:   deque[int]   = deque(maxlen=500)
        self._ep_chunks:    deque[int]   = deque(maxlen=500)
        self._ep_distances: deque[float] = deque(maxlen=500)
        self._total_episodes = 0
        self._stop_requested = False
        self._paused = False

        self._consecutive_timeouts = 0
        self._best_spawn_distance = 0.0
        self._session_deaths = 0
        self._session_timeouts = 0
        self._session_kills = 0
        self._current_ep_start_step = 0

        os.makedirs("data", exist_ok=True)
        os.makedirs("data/hall_of_fame", exist_ok=True)
        self._csv_path = "data/episode_history.csv"
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "global_step", "episode", "reward", "length", "visited_chunks",
                    "max_spawn_distance", "max_depth", "max_x", "kills", 
                    "total_damage", "run_time_s", "death_reason"
                ])

        # State capture for overlay
        self._latest_overlay_stats = ""

        # Register Telegram commands
        notifier.register_stats_provider(self._stats_text)
        notifier.register_command("checkpoint", self._cmd_checkpoint, "Download latest model (.zip)")
        notifier.register_command("mute",       self._cmd_mute,       "Toggle periodic training updates")
        notifier.register_command("pause",      self._cmd_pause,      "Pause Python training script")
        notifier.register_command("lr",         self._cmd_lr,         "Change learning rate (e.g. /lr 0.0001)")
        notifier.register_command("stop",       self._cmd_stop,       "Stop training gracefully")
        notifier.register_command("plot",       self._cmd_plot,       "Send reward and episode length plot")
        notifier.register_command("best",       self._cmd_best,       "Show best runs (Hall of Fame)")
        notifier.register_command("sysinfo",    self._cmd_sysinfo,    "Show system CPU and RAM usage")
        notifier.register_command("logs",   self._cmd_logs,   "Tail the latest run logs")
        notifier.register_command("record", self._cmd_record, "Record ~15s clip of current gameplay")
        notifier.register_command("debug",  self._cmd_debug,  "Show current reward breakdown & recent episode stats")
        notifier.register_command("help",   self._cmd_help,   "Show help message")

    # ── SB3 lifecycle ─────────────────────────────────────────────────────────

    def _on_training_start(self) -> None:
        logger.info("Training started — {} steps, {} envs", self._cfg.total_timesteps, self._cfg.n_envs)
        self._tg.send_text(
            f"🚀 <b>NoitaRL training started</b>\n"
            f"Steps: {self._cfg.total_timesteps:,}\n"
            f"Envs: {self._cfg.n_envs}\n"
            f"LR: {self._cfg.learning_rate}\n"
            f"Hint: Use the Menu button or keyboard below to interact."
        )

        # Override default /screen to include overlay
        def _screen():
            png = self._tg.capture_noita_screen(overlay_text=self._latest_overlay_stats)
            if png:
                self._tg.send_photo(png, caption="📸 Capture from Noita window")
            else:
                self._tg.send_text("⚠️ Could not capture Noita window. Is the game running?")
        self._tg.register_command("screen", _screen, "Capture Noita game screenshot with stats")

    def _on_step(self) -> bool:
        if self._stop_requested:
            logger.warning("Stop requested via Telegram — ending training")
            self._tg.send_text("⛔ Training stopped on your request.")
            return False  # signals SB3 to stop

        while self._paused:
            time.sleep(1.0)
            if self._stop_requested:
                return False

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

            max_x = float(info.get("noita/max_x", 0.0))
            kills         = int(info.get("noita/kills", 0))
            chests_opened = int(info.get("noita/chests_opened", 0))
            total_damage  = float(info.get("noita/total_damage", 0.0))
            run_time = float(info.get("noita/run_time_s", 0.0))
            death_reason = info.get("noita/death_reason", "UNKNOWN")
            
            screenshot = info.get("noita/screenshot")
            route_x = info.get("noita/route_x", [])
            route_y = info.get("noita/route_y", [])
            visually_stuck = info.get("noita/visually_stuck", False)
            action_loop = info.get("noita/action_loop", False)

            self._ep_rewards.append(r)
            self._ep_lengths.append(l)
            self._ep_chunks.append(chunks)
            self._ep_distances.append(dist)
            self._ep_depths.append(depth)
            self._total_episodes += 1
            self._current_ep_start_step = self.num_timesteps

            self._session_kills += kills
            if death_reason == "DEAD":
                self._session_deaths += 1
                self._consecutive_timeouts = 0
            elif death_reason == "TRUNC":
                self._session_timeouts += 1
                self._consecutive_timeouts += 1

            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps, self._total_episodes, r, l, chunks,
                    dist, depth, max_x, kills, total_damage, run_time, death_reason
                ])

            stats_str = f"Ep {self._total_episodes} | {dist:.0f}px | {kills} kills | {death_reason}"
            if screenshot:
                # Save screenshot temporarily in case it's a record or we need to send it
                postcard = TelegramNotifier.make_death_postcard(screenshot, stats_str)
            else:
                postcard = b""

            if dist > self._best_spawn_distance:
                self._best_spawn_distance = dist
                self._on_new_record(dist, depth, kills, route_x, route_y, postcard)
                # Trigger VideoRecorder clip for new record (main process, good quality)
                if self._recorder is not None:
                    rec_ctx = {
                        "dist": dist, "depth": depth, "kills": kills,
                        "steps": l, "reward": r,
                        "episode": self._total_episodes, "chunks": chunks,
                    }
                    self._recorder.force_trigger("new_distance_record", rec_ctx)

            # Check alerts
            if visually_stuck:
                self._tg.send_photo(postcard, caption="⚠️ <b>Agent is visually stuck!</b>\nCoordinates barely changed for 200 steps.")
            if action_loop:
                self._tg.send_photo(postcard, caption="⚠️ <b>Agent in action loop!</b>\nDoing the same action for >80% of the last 500 steps.")

            # ── Episode-level VideoRecorder triggers ──────────────────────────
            if self._recorder is not None:
                rec = self._recorder
                ctx = {
                    "dist":    dist,
                    "depth":   depth,
                    "kills":   kills,
                    "steps":   l,
                    "reward":  r,
                    "episode": self._total_episodes,
                    "chunks":  chunks,
                }
                # New distance record
                # new_distance_record: force_trigger already fired above (_best_spawn_distance check)
                # New depth record (simple check — compare to last 5 ep max)
                if len(self._ep_depths) == 0 or depth > max(list(self._ep_depths)[-5:] or [0]):
                    if depth > 500:   # only meaningful depth
                        rec.trigger_event("new_depth_record", ctx)
                # Kill spree
                if kills >= 3:
                    rec.trigger_event("kill_spree", ctx)
                # Death after long run — high threshold so routine deaths don't spam
                if death_reason == "DEAD" and l >= 600:
                    rec.trigger_event("death_long_run", ctx)
                # Visually stuck
                if visually_stuck:
                    rec.trigger_event("visually_stuck", ctx)
                # Action loop
                if action_loop:
                    rec.trigger_event("action_loop", ctx)
                # Long survival: per-step trigger in noita_env fires at step 600
                # DURING the episode (captures real gameplay, not respawn screen).
                # Here we keep only the high-reward variant for extra-long episodes.
                # High episode reward
                if r > 100.0:
                    rec.trigger_event("high_episode_reward", ctx)

            if self._consecutive_timeouts >= 5:
                self._tg.send_text("⚠️ <b>Agent stuck!</b>\n5 consecutive timeouts without progress.")
                self._consecutive_timeouts = 0

            # Log every episode to loguru
            logger.debug(
                "ep={} reward={:.2f} length={} chunks={} dist={:.0f} depth={:.0f} kills={} reason={} steps={}",
                self._total_episodes, r, l, chunks, dist, depth, kills, death_reason, self.num_timesteps,
            )

            # Surface in SB3's "rollout/" namespace for TensorBoard
            self.logger.record("noita/visited_chunks",     chunks)
            self.logger.record("noita/max_spawn_distance", dist)
            self.logger.record("noita/max_depth",          depth)
            self.logger.record("noita/max_x",              max_x)
            self.logger.record("noita/kills",              kills)
            self.logger.record("noita/chests_opened",      chests_opened)
            self.logger.record("noita/total_damage",       total_damage)

            # W&B per-episode
            if self._cfg.wandb_enabled and _WANDB_OK and wandb.run is not None:
                wandb.log({
                    "episode/reward":         r,
                    "episode/length":         l,
                    "episode/visited_chunks": chunks,
                    "episode/spawn_distance": dist,
                    "episode/max_depth":      depth,
                    "episode/kills":          kills,
                    "episode/total_damage":   total_damage,
                    "episode/run_time_s":     run_time,
                    "episode/total":          self._total_episodes,
                }, step=self.num_timesteps)

        # Periodic Telegram update
        steps_since = self.num_timesteps - self._last_tg
        if steps_since >= self._cfg.telegram_notify_every:
            self._last_tg = self.num_timesteps
            if not self._muted:
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
        if self._cfg.wandb_enabled and _WANDB_OK and wandb.run is not None:
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

    def _on_new_record(self, dist: float, depth: float, kills: int, rx: list, ry: list, postcard: bytes) -> None:
        logger.info("🏆 New Record! Distance: {:.0f} | Depth: {:.0f} | Kills: {}", dist, depth, kills)
        
        # Save to Hall of Fame
        if postcard:
            fname = f"data/hall_of_fame/run_{int(dist)}px_ep{self._total_episodes}.png"
            with open(fname, "wb") as f:
                f.write(postcard)

        caption = (
            f"🏆 <b>New Record Run!</b>\n"
            f"Max Spawn Distance: {dist:.0f} px\n"
            f"Max Depth: {depth:.0f} px\n"
            f"Kills: {kills}"
        )
        if postcard:
            self._tg.send_photo(postcard, caption=caption)
        else:
            self._tg.send_text(caption)
            
        if rx and ry:
            route_png = TelegramNotifier.make_route_plot(rx, ry, title=f"Record Route: {dist:.0f}px")
            self._tg.send_photo(route_png, caption="GPS Track")

    def _stats_text(self) -> str:
        elapsed = time.time() - self._start_ts
        pct = self.num_timesteps / max(self._cfg.total_timesteps, 1) * 100
        mean_r = float(np.mean(self._ep_rewards))   if self._ep_rewards   else 0.0
        mean_l = float(np.mean(self._ep_lengths))   if self._ep_lengths   else 0.0
        mean_c = float(np.mean(self._ep_chunks))    if self._ep_chunks    else 0.0
        mean_d = float(np.mean(self._ep_distances)) if self._ep_distances else 0.0
        sps = self.num_timesteps / max(elapsed, 1)
        
        current_ep_length = self.num_timesteps - self._current_ep_start_step
        
        return (
            f"Steps: {self.num_timesteps:,} / {self._cfg.total_timesteps:,} ({pct:.1f}%)\n"
            f"Episodes: {self._total_episodes:,}\n"
            f"Current Ep Steps: {current_ep_length:,}\n"
            f"Mean reward: {mean_r:.2f}\n"
            f"Mean ep length: {mean_l:.0f}\n"
            f"Mean spawn distance: {mean_d:.0f}\n"
            f"Best distance: {self._best_spawn_distance:.0f}\n"
            f"Session Kills: {self._session_kills:,}\n"
            f"Deaths/Timeouts: {self._session_deaths} / {self._session_timeouts}\n"
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

    def _cmd_record(self) -> None:
        if self._recorder is None:
            self._tg.send_text("⚠️ VideoRecorder is not running.")
            return
        if not self._recorder.is_idle:
            self._tg.send_text("⏳ Recorder is busy — try again in ~20 seconds.")
            return
        ctx = {
            "episode": self._total_episodes,
            "steps":   self._ep_lengths[-1]   if self._ep_lengths   else 0,
            "reward":  self._ep_rewards[-1]   if self._ep_rewards   else 0.0,
            "dist":    self._best_spawn_distance,
            "depth":   float(max(self._ep_depths))   if self._ep_depths   else 0.0,
            "kills":   self._session_kills,
            "chunks":  self._ep_chunks[-1]    if self._ep_chunks    else 0,
        }
        self._recorder.force_trigger("manual_record", ctx)
        self._tg.send_text(
            "🎥 <b>Recording started!</b>\n"
            "Capturing 5s pre-roll + 5s live + 5s post.\n"
            "Clip will arrive in ~15 seconds."
        )

    def _cmd_debug(self) -> None:
        elapsed = time.time() - self._start_ts
        current_ep_steps = self.num_timesteps - self._current_ep_start_step

        last5r = list(self._ep_rewards)[-5:] if self._ep_rewards else []
        last5d = list(self._ep_depths)[-5:]  if self._ep_depths  else []
        last5l = list(self._ep_lengths)[-5:] if self._ep_lengths else []

        r_str = " ".join(f"{v:.1f}" for v in last5r) or "–"
        d_str = " ".join(f"{v:.0f}" for v in last5d) or "–"
        l_str = " ".join(str(v)    for v in last5l) or "–"

        msg = (
            f"🔬 <b>Debug snapshot</b>\n\n"
            f"Global step: {self.num_timesteps:,}\n"
            f"Total episodes: {self._total_episodes:,}\n"
            f"Current ep steps: {current_ep_steps:,}\n"
            f"Session kills: {self._session_kills:,}\n"
            f"Deaths / Timeouts: {self._session_deaths} / {self._session_timeouts}\n"
            f"Best distance: {self._best_spawn_distance:.0f} px\n\n"
            f"<b>Last 5 ep rewards:</b> {r_str}\n"
            f"<b>Last 5 ep depths:</b>  {d_str}\n"
            f"<b>Last 5 ep lengths:</b> {l_str}\n\n"
            f"<i>Tip: set log level to DEBUG in train.py to see per-step reward breakdown.</i>"
        )
        self._tg.send_text(msg)

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

    def _cmd_best(self) -> None:
        files = glob.glob("data/hall_of_fame/run_*.png")
        if not files:
            self._tg.send_text("No runs in the Hall of Fame yet.")
            return
        
        # Sort by distance (extracted from filename run_XXXXpx_epYYY.png)
        def get_dist(f):
            try:
                return int(os.path.basename(f).split('_')[1].replace('px', ''))
            except:
                return 0
        
        files.sort(key=get_dist, reverse=True)
        best_file = files[0]
        
        top_list = "\n".join([f"🏅 {os.path.basename(f).replace('.png', '')}" for f in files[:5]])
        
        with open(best_file, "rb") as f:
            best_png = f.read()
        
        self._tg.send_photo(best_png, caption=f"🌟 <b>Hall of Fame (Top 5)</b>\n\n{top_list}")

    def _cmd_sysinfo(self) -> None:
        try:
            import psutil
            process = psutil.Process(os.getpid())
            ram_mb = process.memory_info().rss / 1024 / 1024
            cpu_pct = process.cpu_percent(interval=0.2)
            sys_ram = psutil.virtual_memory()
            sys_cpu = psutil.cpu_percent(interval=0.2)

            bar_cpu = "█" * int(sys_cpu / 10) + "░" * (10 - int(sys_cpu / 10))
            bar_ram = "█" * int(sys_ram.percent / 10) + "░" * (10 - int(sys_ram.percent / 10))

            text = (
                f"🖥 <b>System Info</b>\n\n"
                f"<b>Global OS:</b>\n"
                f"CPU: [{bar_cpu}] {sys_cpu}%\n"
                f"RAM: [{bar_ram}] {sys_ram.percent}%\n\n"
                f"<b>Agent Process:</b>\n"
                f"CPU: {cpu_pct}%\n"
                f"RAM: {ram_mb:.0f} MB"
            )
            self._tg.send_text(text)
        except Exception as e:
            self._tg.send_text(f"⚠️ Failed to get system info: {e}")

    def _cmd_logs(self) -> None:
        try:
            log_files = glob.glob(os.path.join(self._cfg.log_dir, "*.log"))
            if not log_files:
                self._tg.send_text("⚠️ No log files found.")
                return
            
            latest_log = max(log_files, key=os.path.getmtime)
            
            with open(latest_log, "r", encoding="utf-8") as f:
                # Read last ~20 lines quickly
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                f.seek(max(file_size - 4000, 0), os.SEEK_SET) # Read last 4KB
                lines = f.readlines()
                
            last_lines = "".join(lines[-20:])
            self._tg.send_text(f"📄 <b>Latest Logs:</b> <code>{os.path.basename(latest_log)}</code>\n\n<pre>{last_lines}</pre>")
        except Exception as e:
            self._tg.send_text(f"⚠️ Failed to read logs: {e}")

    def _cmd_help(self) -> None:
        self._tg.send_text(
            "🤖 <b>NoitaRL Bot commands</b>\n"
            "/status — current training stats\n"
            "/plot   — send reward graph\n"
            "/best   — show hall of fame\n"
            "/screen — capture Noita window screenshot\n"
            "/sysinfo — show system CPU and RAM usage\n"
            "/logs   — tail the latest run logs\n"
            "/checkpoint — download latest model .zip\n"
            "/mute   — toggle periodic updates\n"
            "/stop   — stop training gracefully\n"
            "/help   — this message"
        )

    def _cmd_checkpoint(self) -> None:
        try:
            zips = glob.glob(os.path.join(self._cfg.checkpoint_dir, "*.zip"))
            if not zips:
                self._tg.send_text("⚠️ No checkpoints found.")
                return
            latest_zip = max(zips, key=os.path.getmtime)
            size_mb = os.path.getsize(latest_zip) / (1024 * 1024)
            self._tg.send_document(
                latest_zip,
                caption=f"📦 <b>Latest Checkpoint</b>\n{os.path.basename(latest_zip)} ({size_mb:.1f} MB)"
            )
        except Exception as e:
            self._tg.send_text(f"⚠️ Failed to send checkpoint: {e}")

    def _cmd_mute(self) -> None:
        self._muted = not self._muted
        if self._muted:
            self._tg.send_text("🔕 <b>Muted</b>\nPeriodic progress updates paused. Use /status to check manually.")
        else:
            self._tg.send_text("🔔 <b>Unmuted</b>\nPeriodic progress updates resumed.")

    def _cmd_pause(self) -> None:
        self._paused = not self._paused
        if self._paused:
            self._tg.send_text("⏸ <b>Training Paused</b>\nPython bridge stopped. Noita physics will run idle. Press ESC in Noita to pause the engine. Use /pause again to resume.")
        else:
            self._tg.send_text("▶️ <b>Training Resumed</b>")

    def _cmd_lr(self, text: str) -> None:
        try:
            parts = text.split()
            if len(parts) < 2:
                raise ValueError("Missing value")
            val = float(parts[1])
            if hasattr(self.model, "policy") and hasattr(self.model.policy, "optimizer"):
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group['lr'] = val
            self.model.learning_rate = val
            self._tg.send_text(f"✅ Learning rate changed to <b>{val}</b>")
            logger.info("Learning rate changed to {}", val)
        except Exception:
            self._tg.send_text("⚠️ Usage: <code>/lr 0.0001</code>")
