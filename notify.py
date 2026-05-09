"""
Lightweight Telegram notifier — no extra library, pure requests.

Features:
  • send_text()      — plain / HTML message
  • send_photo()     — PNG bytes (e.g. matplotlib plot)
  • Command polling  — /status /stop /plot in a background thread
  • Graceful no-op   — if token / chat_id are empty, every method silently returns
"""

from __future__ import annotations

import io
import threading
import time
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from loguru import logger


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self._token    = token
        self._chat_id  = chat_id
        self._enabled  = bool(token and chat_id)
        self._base     = f"https://api.telegram.org/bot{token}"
        self._offset   = 0
        self._running  = False
        self._handlers: dict[str, Callable] = {}
        self._stats_fn: Optional[Callable[[], str]] = None  # injected by callback

        if self._enabled:
            logger.info("Telegram notifier initialised (chat_id={})", chat_id)

    # ── Sending ───────────────────────────────────────────────────────────────

    def send_text(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self._enabled:
            return False
        try:
            r = requests.post(
                f"{self._base}/sendMessage",
                json={"chat_id": self._chat_id, "text": text, "parse_mode": parse_mode},
                timeout=10,
            )
            if r.status_code != 200:
                logger.warning("Telegram sendMessage returned {}: {}", r.status_code, r.text[:200])
            return r.status_code == 200
        except Exception as exc:
            logger.warning("Telegram send_text failed: {}", exc)
            return False

    def send_photo(self, png_bytes: bytes, caption: str = "") -> bool:
        if not self._enabled:
            return False
        try:
            r = requests.post(
                f"{self._base}/sendPhoto",
                files={"photo": ("plot.png", png_bytes, "image/png")},
                data={"chat_id": self._chat_id, "caption": caption},
                timeout=20,
            )
            return r.status_code == 200
        except Exception as exc:
            logger.warning("Telegram send_photo failed: {}", exc)
            return False

    # ── Plots ─────────────────────────────────────────────────────────────────

    @staticmethod
    def make_reward_plot(rewards: list[float], title: str = "NoitaRL — Episode rewards") -> bytes:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rewards, alpha=0.5, linewidth=0.8, label="raw")
        if len(rewards) >= 20:
            import numpy as np
            window = max(10, len(rewards) // 20)
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(rewards)), ma, linewidth=2, label=f"MA-{window}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    # ── Command bot ───────────────────────────────────────────────────────────

    def register_command(self, command: str, handler: Callable) -> None:
        """Register a callable to be invoked when /command is received."""
        self._handlers[f"/{command}"] = handler

    def register_stats_provider(self, fn: Callable[[], str]) -> None:
        """fn() should return a formatted string of current training stats."""
        self._stats_fn = fn

    def start_polling(self) -> None:
        if not self._enabled:
            return
        self._running = True

        # Register built-in /status command
        if "/status" not in self._handlers:
            def _status():
                text = self._stats_fn() if self._stats_fn else "No stats available yet."
                self.send_text(f"📊 <b>Status</b>\n{text}")
            self._handlers["/status"] = _status

        t = threading.Thread(target=self._poll_loop, daemon=True, name="telegram-poll")
        t.start()
        logger.info("Telegram command polling started")

    def stop(self) -> None:
        self._running = False

    def _poll_loop(self) -> None:
        while self._running:
            try:
                resp = requests.get(
                    f"{self._base}/getUpdates",
                    params={"offset": self._offset, "timeout": 10},
                    timeout=15,
                ).json()
                for update in resp.get("result", []):
                    self._offset = update["update_id"] + 1
                    text = update.get("message", {}).get("text", "").strip()
                    handler = self._handlers.get(text)
                    if handler:
                        try:
                            handler()
                        except Exception as exc:
                            logger.warning("Telegram handler '{}' raised: {}", text, exc)
            except Exception as exc:
                logger.debug("Telegram poll error (will retry): {}", exc)
            time.sleep(2)
