"""
VideoRecorder — event-triggered highlight recorder for NoitaRL.

Architecture:
  • Background capture thread grabs Noita window at CAPTURE_FPS.
  • Rolling PRE_SEC-second ring buffer always fills.
  • When trigger_event() fires:
      pre-buffer snapshot → live recording for up to RECORD_SEC → POST_SEC tail
  • Clip assembled as GIF → Groq LLaMA generates description → Telegram.
  • Local copy saved to data/highlights/.

Trigger cooldown: COOLDOWN_SEC between recordings (prevents spam).

Extreme events detected (15 types):
  portal_teleport      — Δpos > 300 px while near portal
  new_distance_record  — furthest ever from spawn
  new_depth_record     — deepest ever
  kill_spree           — 3+ kills in one episode
  instant_damage       — HP drops ≥50% in one step
  death_long_run       — dies after 300+ steps
  fast_movement        — |vx|>120 sustained for 15 steps
  extreme_fall         — vy > 350 (free-fall into shaft)
  visually_stuck       — < 20×20 px movement over 200 steps
  action_loop          — same action 80%+ of last 500 steps
  chunk_burst          — 5+ new chunks in 20 steps
  reward_spike         — single-step reward > 15.0
  long_survival        — episode ≥ 600 steps without dying
  high_episode_reward  — total episode reward > 100.0
  wand_kill            — wand shot killed an enemy
"""

from __future__ import annotations

import collections
import io
import os
import queue
import threading
import time
from datetime import datetime
from typing import Optional

from loguru import logger
from PIL import Image


HASHTAGS = [
    "#BotGaming",
    "#EpicMove",
    "#CloseCall",
    "#KillSpree",
    "#Exploring",
    "#DeepDive",
    "#Speedrun",
    "#Oops",
    "#Stuck",
    "#Milestone",
]

_EVENT_HASHTAG: dict[str, str] = {
    "portal_teleport":      "#EpicMove",
    "new_distance_record":  "#Milestone",
    "new_depth_record":     "#DeepDive",
    "kill_spree":           "#KillSpree",
    "instant_damage":       "#CloseCall",
    "death_long_run":       "#Oops",
    "fast_movement":        "#Speedrun",
    "extreme_fall":         "#Speedrun",
    "visually_stuck":       "#Stuck",
    "action_loop":          "#Stuck",
    "chunk_burst":          "#Exploring",
    "reward_spike":         "#EpicMove",
    "long_survival":        "#BotGaming",
    "high_episode_reward":  "#Milestone",
    "wand_kill":            "#KillSpree",
}

_DESCRIPTIONS: dict[str, str] = {
    "portal_teleport":      "Agent found a Holy Mountain portal and teleported to the next level! dist={dist:.0f}px",
    "new_distance_record":  "New distance record: {dist:.0f}px from spawn! depth={depth:.0f}px",
    "new_depth_record":     "New depth record: {depth:.0f}px underground! ep={episode}",
    "kill_spree":           "{kills} enemies eliminated in this episode! ep={episode}",
    "instant_damage":       "Agent took massive damage — HP dropped {damage:.0%} in one step!",
    "death_long_run":       "Long run of {steps} steps ended in death. reward={reward:.1f}",
    "fast_movement":        "Agent flying at extreme horizontal speed: vx={vx:.0f}px/s",
    "extreme_fall":         "Agent in free-fall: vy={vy:.0f}px/s plummeting into the depths!",
    "visually_stuck":       "Agent stuck — moved less than 20px for 200 consecutive steps.",
    "action_loop":          "Agent looping — repeating the same action 80%+ of last 500 steps.",
    "chunk_burst":          "Exploration burst: {chunks} new 32×32 chunks discovered in 20 steps!",
    "reward_spike":         "Massive reward spike: +{reward:.1f} in a single step!",
    "long_survival":        "Survival milestone! Episode reached {steps} steps. reward={reward:.1f}",
    "high_episode_reward":  "High episode reward: {reward:.1f}! chunks={chunks} kills={kills}",
    "wand_kill":            "Direct hit! Wand killed an enemy. total kills this ep: {kills}",
}


class VideoRecorder:
    """
    Background video recorder for extreme events.

    Usage:
        recorder = VideoRecorder(notifier, groq_api_key="...")
        recorder.start()
        ...
        recorder.trigger_event("portal_teleport", {"dist": 1234, "depth": 567})
        ...
        recorder.stop()
    """

    CAPTURE_FPS   = 10          # frames per second captured
    PRE_SEC       = 5           # seconds of pre-event footage to keep
    RECORD_SEC    = 5           # max seconds of live recording per event
    POST_SEC      = 5           # seconds of post-event footage to append
    COOLDOWN_SEC  = 20          # min seconds between recordings
    FRAME_W       = 480         # rescale width  (height is auto)
    FRAME_H       = 300         # rescale height
    SAVE_DIR      = "data/highlights"

    def __init__(self, notifier, groq_api_key: str = ""):
        self._notifier   = notifier
        self._groq_key   = groq_api_key.strip()

        pre_len = self.CAPTURE_FPS * self.PRE_SEC
        self._pre_buf: collections.deque = collections.deque(maxlen=pre_len)
        self._live_frames: list           = []
        self._post_frames: list           = []

        # State machine: "idle" | "recording" | "post" | "cooldown"
        self._state       = "idle"
        self._state_lock  = threading.Lock()
        self._active_evt: Optional[dict]  = None
        self._record_start = 0.0
        self._last_sent    = 0.0

        self._event_q: queue.Queue = queue.Queue()
        self._window_bounds: Optional[dict] = None

        self._running = False
        self._threads: list[threading.Thread] = []

        os.makedirs(self.SAVE_DIR, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        for name, target in [
            ("vid-capture", self._capture_loop),
            ("vid-events",  self._event_loop),
        ]:
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)
        logger.info(
            "VideoRecorder ready — {}fps, {}s pre+{}s live+{}s post, cooldown {}s",
            self.CAPTURE_FPS, self.PRE_SEC, self.RECORD_SEC, self.POST_SEC, self.COOLDOWN_SEC,
        )

    def stop(self) -> None:
        self._running = False

    def trigger_event(self, event_name: str, context: dict) -> None:
        """Thread-safe call from env or callback thread."""
        now = time.monotonic()
        if now - self._last_sent < self.COOLDOWN_SEC:
            return
        self._event_q.put_nowait({"name": event_name, "ctx": context, "t": now})

    # ── Capture thread ────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        interval = 1.0 / self.CAPTURE_FPS
        while self._running:
            t0 = time.monotonic()
            frame = self._grab_frame()
            if frame is not None:
                with self._state_lock:
                    st = self._state
                if st == "idle" or st == "cooldown":
                    self._pre_buf.append(frame)
                elif st == "recording":
                    with self._state_lock:
                        self._live_frames.append(frame)
                elif st == "post":
                    with self._state_lock:
                        self._post_frames.append(frame)
            sleep = interval - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def _grab_frame(self) -> Optional[Image.Image]:
        try:
            import mss
            bounds = self._window_bounds
            if bounds is None:
                bounds = self._find_noita_window()
                if bounds is None:
                    return None
                self._window_bounds = bounds
            with mss.mss() as sct:
                raw = sct.grab(bounds)
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                return img.resize((self.FRAME_W, self.FRAME_H), Image.BILINEAR)
        except Exception:
            self._window_bounds = None
            return None

    @staticmethod
    def _find_noita_window() -> Optional[dict]:
        try:
            import pygetwindow as gw
            wins = gw.getWindowsWithTitle("Noita")
            if not wins:
                return None
            w = wins[0]
            if w.width < 50 or w.height < 50:
                return None
            return {
                "top":    max(0, w.top),
                "left":   max(0, w.left),
                "width":  w.width,
                "height": w.height,
            }
        except Exception:
            return None

    # ── Event loop ────────────────────────────────────────────────────────────

    def _event_loop(self) -> None:
        """Drive state machine: idle → recording → post → idle."""
        while self._running:
            try:
                evt = self._event_q.get(timeout=0.25)
            except queue.Empty:
                # Advance state machine if needed
                self._tick_state()
                continue

            with self._state_lock:
                st = self._state
                if st in ("cooldown", "post"):
                    logger.debug("VideoRecorder: '{}' skipped ({})", evt["name"], st)
                    continue
                if st == "recording":
                    # Extend recording window on new event
                    self._record_start = time.monotonic()
                    logger.debug("VideoRecorder: '{}' extends recording", evt["name"])
                    continue
                # idle → recording
                self._active_evt   = evt
                self._record_start = time.monotonic()
                self._live_frames  = []
                self._post_frames  = []
                self._state        = "recording"
            logger.info("VideoRecorder: recording triggered by '{}'", evt["name"])

    def _tick_state(self) -> None:
        with self._state_lock:
            st = self._state
            if st != "recording":
                return
            elapsed = time.monotonic() - self._record_start
            if elapsed < self.RECORD_SEC:
                return
            # recording → post
            self._state = "post"
            evt        = self._active_evt
            pre        = list(self._pre_buf)
            live       = list(self._live_frames)

        logger.debug("VideoRecorder: recording done, collecting {}s post-footage", self.POST_SEC)
        # Collect post frames in this thread (state == "post" tells capture to write there)
        time.sleep(self.POST_SEC)
        with self._state_lock:
            post      = list(self._post_frames)
            self._state = "cooldown"
            self._last_sent = time.monotonic()

        threading.Thread(
            target=self._assemble_and_send,
            args=(evt, pre, live, post),
            daemon=True,
            name="vid-assemble",
        ).start()

        # Cooldown window, then back to idle
        time.sleep(self.COOLDOWN_SEC)
        with self._state_lock:
            if self._state == "cooldown":
                self._state = "idle"

    # ── Assembly & dispatch ───────────────────────────────────────────────────

    def _assemble_and_send(
        self,
        evt: dict,
        pre: list,
        live: list,
        post: list,
    ) -> None:
        all_frames = pre + live + post
        if len(all_frames) < 3:
            logger.warning("VideoRecorder: too few frames ({}), skipping", len(all_frames))
            return

        event_name = evt.get("name", "event")
        context    = evt.get("ctx", {})
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path   = os.path.join(self.SAVE_DIR, f"{event_name}_{ts}.gif")

        logger.info(
            "VideoRecorder: assembling {} frames ({} pre + {} live + {} post)",
            len(all_frames), len(pre), len(live), len(post),
        )

        try:
            gif_bytes = self._make_gif(all_frames)
        except Exception as exc:
            logger.error("VideoRecorder: GIF creation failed: {}", exc)
            return

        # Save locally
        try:
            with open(gif_path, "wb") as f:
                f.write(gif_bytes)
            logger.info("VideoRecorder: saved {} ({:.1f} MB)", gif_path, len(gif_bytes) / 1e6)
        except Exception as exc:
            logger.warning("VideoRecorder: save failed: {}", exc)

        # AI caption
        description = self._groq_describe(event_name, context)
        hashtag     = _EVENT_HASHTAG.get(event_name, "#BotGaming")
        caption = (
            f"🎥 <b>{event_name.replace('_', ' ').title()}</b>\n"
            f"{description}\n"
            f"<i>{hashtag}</i>"
        )

        # Telegram — try sendAnimation first (plays inline), fall back to document
        try:
            self._notifier.send_animation(gif_bytes, caption=caption)
        except Exception:
            try:
                self._notifier.send_document(gif_path, caption=caption)
            except Exception as exc2:
                logger.warning("VideoRecorder: TG send failed: {}", exc2)

    def _make_gif(self, frames: list) -> bytes:
        buf  = io.BytesIO()
        dur  = max(50, int(1000 / self.CAPTURE_FPS))
        frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=dur,
            loop=0,
            optimize=True,
        )
        return buf.getvalue()

    @staticmethod
    def _read_recent_logs(n_lines: int = 30) -> str:
        """Read the last N lines from logger.txt (Noita mod log)."""
        log_path = os.path.join(os.path.dirname(__file__), "logger.txt")
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return "".join(lines[-n_lines:]).strip()
        except Exception:
            return ""

    @staticmethod
    def _read_recent_actions(n: int = 20) -> str:
        """Read the last N lines from actions_trace.jsonl."""
        trace_path = os.path.join(os.path.dirname(__file__), "actions_trace.jsonl")
        try:
            with open(trace_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return "".join(lines[-n:]).strip()
        except Exception:
            return ""

    def _groq_describe(self, event_name: str, ctx: dict) -> str:
        template = _DESCRIPTIONS.get(event_name, "Extreme event: {name}")
        fallback = template.format(name=event_name, **{
            k: ctx.get(k, 0.0) for k in (
                "dist", "depth", "kills", "damage", "steps", "reward",
                "chunks", "vx", "vy", "episode",
            )
        })

        if not self._groq_key:
            return fallback

        try:
            from groq import Groq
            tag_list     = ", ".join(HASHTAGS)
            noita_logs   = self._read_recent_logs(30)
            action_trace = self._read_recent_actions(20)

            prompt = (
                "You are an AI analyst watching a Reinforcement Learning bot (PPO agent) "
                "learn to play Noita — a roguelite physics-based dungeon crawler. "
                "The bot controls the player using direct velocity injection, learns from "
                "chunk-exploration rewards, kill bonuses, and depth progress.\n\n"

                f"== TRIGGERED EVENT ==\n"
                f"Type: {event_name}\n"
                f"Stats: episode={ctx.get('episode','?')} steps={ctx.get('steps','?')} "
                f"dist={ctx.get('dist',0):.0f}px depth={ctx.get('depth',0):.0f}px "
                f"kills={ctx.get('kills','?')} reward={ctx.get('reward',0):.1f} "
                f"chunks={ctx.get('chunks','?')}\n\n"

                f"== LAST 30 LINES FROM NOITA MOD LOG ==\n{noita_logs}\n\n"

                f"== LAST 20 ACTION TRACE ENTRIES (JSON) ==\n{action_trace}\n\n"

                "Write a SHORT (2-3 sentences) analysis of what happened and why it's "
                "interesting from an RL/training perspective. "
                "Comment on what the agent likely learned or is struggling with. "
                "Be specific — reference actual numbers from the logs. "
                f"End with ONE hashtag from: {tag_list}. "
                "Format: <analysis>\\n<hashtag>. Plain text only."
            )

            resp = Groq(api_key=self._groq_key).chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_completion_tokens=120,
                top_p=1,
                stream=False,
            )
            text = resp.choices[0].message.content.strip()
            return text if any(h in text for h in HASHTAGS) else (
                f"{text}\n{_EVENT_HASHTAG.get(event_name, '#BotGaming')}"
            )
        except Exception as exc:
            logger.debug("VideoRecorder: Groq failed: {}", exc)
            return fallback
