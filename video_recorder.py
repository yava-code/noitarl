from collections import deque
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
    "manual_record":        "#BotGaming",
}

_DESCRIPTIONS: dict[str, str] = {
    "manual_record":        "Manual recording — ep={episode} steps={steps} reward={reward:.1f} dist={dist:.0f}px",
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

    CAPTURE_FPS   = 10          # frames per second (lower → smaller GIF)
    PRE_SEC       = 3           # seconds of pre-event footage to keep
    RECORD_SEC    = 4           # max seconds of live recording per event
    POST_SEC      = 3           # seconds of post-event footage to append
    COOLDOWN_SEC  = 20          # min seconds between recordings
    FRAME_W       = 480         # rescale width  (was 640 → ~12MB GIFs that TG couldn't play)
    FRAME_H       = 300         # rescale height (was 400 → now targets ~3-5MB)
    SAVE_DIR      = "data/highlights"
    MAX_LOCAL_GIFS = 100        # keep only the last N most interesting/recent gifs locally

    def __init__(self, notifier, groq_api_key: str = "", pid: Optional[int] = None):
        self._notifier   = notifier
        self._groq_key   = groq_api_key.strip()
        # When several Noita instances run side-by-side, EnumWindows returns
        # the first match. Pin the recorder to a specific PID so the GIF
        # doesn't ping-pong between two windows.
        self._pid: Optional[int] = pid

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
        self._window_hwnd: Optional[int] = None

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

    def force_trigger(self, event_name: str, context: dict) -> None:
        """Like trigger_event but bypasses the cooldown timer (for /record command)."""
        self._event_q.put_nowait({"name": event_name, "ctx": context, "t": time.monotonic()})

    @property
    def is_idle(self) -> bool:
        """True when the recorder is idle and ready for a new trigger."""
        with self._state_lock:
            return self._state == "idle"

    @property
    def status(self) -> str:
        """Current recorder state: 'idle' | 'recording' | 'post' | 'cooldown'."""
        with self._state_lock:
            return self._state

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
        hwnd = self._window_hwnd
        if hwnd is None:
            hwnd = self._find_noita_hwnd(pid_filter=self._pid)
            if hwnd is None:
                return None
            self._window_hwnd = hwnd
            logger.debug("VideoRecorder: found Noita hwnd={} pid={}", hwnd, self._pid)
        img = self._print_window(hwnd)
        if img is None:
            self._window_hwnd = None
            return None
        return img.resize((self.FRAME_W, self.FRAME_H), Image.LANCZOS)

    @staticmethod
    def _find_noita_hwnd(pid_filter: Optional[int] = None) -> Optional[int]:
        """
        Return HWND of the Noita game window, identified by process executable
        name (noita.exe / noita_dev.exe) — NOT by window title, so browser tabs
        named 'noita-rl-...' are never mistakenly matched.

        If ``pid_filter`` is given, only a window owned by that specific PID is
        accepted. This is required when running multiple Noita instances side
        by side so each env's recorder stays pinned to its own window.
        """
        try:
            import ctypes
            import ctypes.wintypes as wt

            k32   = ctypes.windll.kernel32
            user32 = ctypes.windll.user32
            PROCESS_QUERY_LIMITED = 0x1000

            found: list = []

            @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
            def _cb(hwnd, _):
                if not user32.IsWindowVisible(hwnd):
                    return True
                pid = wt.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if not pid.value:
                    return True
                if pid_filter is not None and pid.value != pid_filter:
                    return True
                hproc = k32.OpenProcess(PROCESS_QUERY_LIMITED, False, pid.value)
                if not hproc:
                    return True
                try:
                    buf  = ctypes.create_unicode_buffer(260)
                    size = ctypes.c_uint32(260)
                    if k32.QueryFullProcessImageNameW(hproc, 0, buf, ctypes.byref(size)):
                        # Match only the exe filename, not the full path or window title
                        exe = buf.value.lower().rsplit("\\", 1)[-1]
                        if exe in ("noita.exe", "noita_dev.exe"):
                            rect = wt.RECT()
                            user32.GetClientRect(hwnd, ctypes.byref(rect))
                            if rect.right >= 50 and rect.bottom >= 50:
                                found.append(int(hwnd))
                                return False  # stop after first match
                finally:
                    k32.CloseHandle(hproc)
                return True

            user32.EnumWindows(_cb, 0)
            return found[0] if found else None
        except Exception:
            return None

    @staticmethod
    def _print_window(hwnd: int) -> Optional[Image.Image]:
        """
        Capture window pixel content via PrintWindow — works even when the
        window is behind other windows or minimized.
        PW_RENDERFULLCONTENT (flag=2) forces DWM to render GPU/OpenGL content.
        """
        try:
            import ctypes
            import ctypes.wintypes as wt

            user32 = ctypes.windll.user32
            gdi32  = ctypes.windll.gdi32

            if not user32.IsWindow(hwnd):
                return None

            rect = wt.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            w, h = rect.right, rect.bottom
            if w <= 0 or h <= 0:
                return None

            hdc_src = user32.GetWindowDC(hwnd)
            hdc_mem = gdi32.CreateCompatibleDC(hdc_src)
            hbmp    = gdi32.CreateCompatibleBitmap(hdc_src, w, h)
            gdi32.SelectObject(hdc_mem, hbmp)

            # Flag 3 = PW_CLIENTONLY(1) | PW_RENDERFULLCONTENT(2)
            user32.PrintWindow(hwnd, hdc_mem, 3)

            class _BIH(ctypes.Structure):
                _fields_ = [
                    ("biSize",          ctypes.c_uint32),
                    ("biWidth",         ctypes.c_int32),
                    ("biHeight",        ctypes.c_int32),
                    ("biPlanes",        ctypes.c_uint16),
                    ("biBitCount",      ctypes.c_uint16),
                    ("biCompression",   ctypes.c_uint32),
                    ("biSizeImage",     ctypes.c_uint32),
                    ("biXPelsPerMeter", ctypes.c_int32),
                    ("biYPelsPerMeter", ctypes.c_int32),
                    ("biClrUsed",       ctypes.c_uint32),
                    ("biClrImportant",  ctypes.c_uint32),
                ]

            bih = _BIH()
            bih.biSize      = ctypes.sizeof(_BIH)
            bih.biWidth     = w
            bih.biHeight    = -h  # negative = top-down
            bih.biPlanes    = 1
            bih.biBitCount  = 32
            bih.biCompression = 0  # BI_RGB

            raw = (ctypes.c_char * (w * h * 4))()
            lines = gdi32.GetDIBits(hdc_mem, hbmp, 0, h, raw, ctypes.byref(bih), 0)

            img = None
            if lines > 0:
                img = Image.frombuffer("RGBA", (w, h), bytes(raw), "raw", "BGRA", 0, 1).convert("RGB")

            gdi32.DeleteObject(hbmp)
            gdi32.DeleteDC(hdc_mem)
            user32.ReleaseDC(hwnd, hdc_src)
            return img
        except Exception as exc:
            logger.debug("VideoRecorder: PrintWindow failed: {}", exc)
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

        # Check frame diversity: if all frames look the same (agent stuck / respawn screen)
        # the GIF will show as a static photo in Telegram. Skip TG send in that case.
        import numpy as _np
        _send_to_tg = True
        if len(all_frames) >= 2:
            try:
                _a = _np.array(all_frames[0]).astype(float)
                _b = _np.array(all_frames[len(all_frames) // 2]).astype(float)
                _frame_diff = _np.mean(_np.abs(_a - _b))
                if _frame_diff < 5.0:
                    logger.info(
                        "VideoRecorder: frames are static (diff={:.2f}), saving locally but skipping TG for '{}'",
                        _frame_diff, event_name,
                    )
                    _send_to_tg = False
            except Exception:
                pass

        try:
            gif_bytes = self._make_gif(all_frames)
        except Exception as exc:
            logger.error("VideoRecorder: GIF creation failed: {}", exc)
            return

        # Too-small GIF = effectively a single frame (e.g. all frames identical after LZW)
        if len(gif_bytes) < 8_000:
            logger.info(
                "VideoRecorder: GIF too small ({} bytes), likely static — saving locally, skipping TG",
                len(gif_bytes),
            )
            _send_to_tg = False

        # Save locally
        try:
            with open(gif_path, "wb") as f:
                f.write(gif_bytes)
            logger.info("VideoRecorder: saved {} ({:.1f} MB)", gif_path, len(gif_bytes) / 1e6)
        except Exception as exc:
            logger.warning("VideoRecorder: save failed: {}", exc)

        if not _send_to_tg:
            self._cleanup_local_storage()
            return

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

        # Local cleanup: keep only the most recent/relevant files
        self._cleanup_local_storage()

    def _cleanup_local_storage(self) -> None:
        """Keep the local folder size under control by deleting older/less interesting highlights."""
        try:
            files = []
            for f in os.listdir(self.SAVE_DIR):
                if not f.endswith(".gif"):
                    continue
                path = os.path.join(self.SAVE_DIR, f)
                files.append((path, os.path.getmtime(path)))

            if len(files) <= self.MAX_LOCAL_GIFS:
                return

            # Sort by modification time (oldest first)
            files.sort(key=lambda x: x[1])

            to_delete = len(files) - self.MAX_LOCAL_GIFS
            for i in range(to_delete):
                try:
                    os.remove(files[i][0])
                except Exception:
                    pass
            logger.info("VideoRecorder: cleaned up {} old local highlights", to_delete)
        except Exception as exc:
            logger.debug("VideoRecorder: cleanup failed: {}", exc)

    def _make_gif(self, frames: list) -> bytes:
        buf = io.BytesIO()
        dur = max(40, int(1000 / self.CAPTURE_FPS))

        # Build a unified palette from a sample of all frames so colours are
        # consistent across the clip (avoids palette-flicker between frames).
        import numpy as np
        pixels = np.concatenate(
            [np.array(f).reshape(-1, 3) for f in frames[::max(1, len(frames)//10)]],
            axis=0,
        )
        idx = np.random.default_rng(0).choice(len(pixels), min(50_000, len(pixels)), replace=False)
        sample_img = Image.fromarray(pixels[idx].reshape(-1, 1, 3).astype("uint8"))
        palette_img = sample_img.quantize(colors=255, method=Image.Quantize.FASTOCTREE)

        # Quantize each frame to the shared palette with Floyd-Steinberg dithering
        quantized = [f.quantize(palette=palette_img, dither=1) for f in frames]

        quantized[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=quantized[1:],
            duration=dur,
            loop=0,
            optimize=False,  # skip slow LZW re-optimisation when using shared palette
        )
        return buf.getvalue()

    @staticmethod
    def _read_recent_logs(n_lines: int = 30) -> str:
        """Read the last N lines from logger.txt (Noita mod log)."""
        log_path = os.path.join(os.path.dirname(__file__), "logger.txt")
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = deque(f, maxlen=n_lines)
            return "".join(lines).strip()
        except Exception:
            return ""

    @staticmethod
    def _read_recent_actions(n: int = 20) -> str:
        """Read the last N lines from actions_trace.jsonl."""
        trace_path = os.path.join(os.path.dirname(__file__), "actions_trace.jsonl")
        try:
            with open(trace_path, "r", encoding="utf-8", errors="replace") as f:
                lines = deque(f, maxlen=n)
            return "".join(lines).strip()
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
        return fallback
