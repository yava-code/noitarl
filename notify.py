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
from PIL import Image, ImageDraw, ImageFont


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self._token    = token
        self._chat_id  = chat_id
        self._enabled  = bool(token and chat_id)
        self._base     = f"https://api.telegram.org/bot{token}"
        self._offset   = 0
        self._running  = False
        self._handlers: dict[str, Callable] = {}
        self._commands_desc: list[dict] = []
        self._stats_fn: Optional[Callable[[], str]] = None  # injected by callback
        self._buttons_row_1 = [
            {"text": "📊 Status", "callback_data": "status"},
            {"text": "📈 Plot", "callback_data": "plot"},
            {"text": "📸 Screen", "callback_data": "screen"}
        ]
        self._buttons_row_2 = [
            {"text": "🏆 Best", "callback_data": "best"},
            {"text": "🖥 SysInfo", "callback_data": "sysinfo"},
            {"text": "📄 Logs", "callback_data": "logs"}
        ]
        self._buttons_row_3 = [
            {"text": "📦 Model", "callback_data": "checkpoint"},
            {"text": "🔕 Mute", "callback_data": "mute"},
            {"text": "⏹ Stop", "callback_data": "stop"}
        ]

        if self._enabled:
            logger.info("Telegram notifier initialised (chat_id={})", chat_id)

    # ── Sending ───────────────────────────────────────────────────────────────

    def send_text(self, text: str, parse_mode: str = "HTML", reply_markup: Optional[dict] = None) -> bool:
        if not self._enabled:
            return False
        try:
            payload = {"chat_id": self._chat_id, "text": text, "parse_mode": parse_mode}
            if reply_markup:
                payload["reply_markup"] = reply_markup
            else:
                # Add default keyboard
                payload["reply_markup"] = {
                    "inline_keyboard": [self._buttons_row_1, self._buttons_row_2, self._buttons_row_3]
                }
            r = requests.post(f"{self._base}/sendMessage", json=payload, timeout=10)
            if r.status_code != 200:
                logger.warning("Telegram sendMessage returned {}: {}", r.status_code, r.text[:200])
            return r.status_code == 200
        except Exception as exc:
            logger.warning("Telegram send_text failed: {}", exc)
            return False

    def send_animation(self, gif_bytes: bytes, caption: str = "") -> bool:
        """Send GIF as inline animation (loops in Telegram chat)."""
        if not self._enabled:
            return False
        try:
            data = {"chat_id": self._chat_id, "caption": caption, "parse_mode": "HTML"}
            r = requests.post(
                f"{self._base}/sendAnimation",
                files={"animation": ("highlight.gif", gif_bytes, "image/gif")},
                data=data,
                timeout=60,
            )
            if r.status_code != 200:
                logger.warning("Telegram sendAnimation returned {}: {}", r.status_code, r.text[:200])
            return r.status_code == 200
        except Exception as exc:
            logger.warning("Telegram send_animation failed: {}", exc)
            return False

    def send_document(self, file_path: str, caption: str = "", reply_markup: Optional[dict] = None) -> bool:
        if not self._enabled:
            return False
        try:
            data = {"chat_id": self._chat_id, "caption": caption}
            if reply_markup is None:
                data["reply_markup"] = '{"inline_keyboard": [[{"text": "📊 Status", "callback_data": "status"}, {"text": "📈 Plot", "callback_data": "plot"}, {"text": "📸 Screen", "callback_data": "screen"}], [{"text": "🏆 Best", "callback_data": "best"}, {"text": "🖥 SysInfo", "callback_data": "sysinfo"}, {"text": "📄 Logs", "callback_data": "logs"}], [{"text": "📦 Model", "callback_data": "checkpoint"}, {"text": "🔕 Mute", "callback_data": "mute"}, {"text": "⏹ Stop", "callback_data": "stop"}]]}'
            elif reply_markup:
                import json
                data["reply_markup"] = json.dumps(reply_markup)

            with open(file_path, "rb") as f:
                r = requests.post(
                    f"{self._base}/sendDocument",
                    data=data,
                    files={"document": f},
                    timeout=60,
                )
            return r.status_code == 200
        except Exception as exc:
            logger.warning("Telegram send_document failed: {}", exc)
            return False

    def send_photo(self, png_bytes: bytes, caption: str = "", reply_markup: Optional[dict] = None) -> bool:
        if not self._enabled:
            return False
        try:
            data = {"chat_id": self._chat_id, "caption": caption}
            if reply_markup is None:
                data["reply_markup"] = '{"inline_keyboard": [[{"text": "📊 Status", "callback_data": "status"}, {"text": "📈 Plot", "callback_data": "plot"}, {"text": "📸 Screen", "callback_data": "screen"}], [{"text": "🏆 Best", "callback_data": "best"}, {"text": "🖥 SysInfo", "callback_data": "sysinfo"}, {"text": "📄 Logs", "callback_data": "logs"}], [{"text": "📦 Model", "callback_data": "checkpoint"}, {"text": "🔕 Mute", "callback_data": "mute"}, {"text": "⏹ Stop", "callback_data": "stop"}]]}'
            elif reply_markup:
                import json
                data["reply_markup"] = json.dumps(reply_markup)

            r = requests.post(
                f"{self._base}/sendPhoto",
                files={"photo": ("plot.png", png_bytes, "image/png")},
                data=data,
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

    @staticmethod
    def make_route_plot(route_x: list[float], route_y: list[float], title: str = "Agent Route") -> bytes:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(route_x, route_y, color='red', alpha=0.7, linewidth=1.5, marker='o', markersize=2, markevery=10)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    @staticmethod
    def make_death_postcard(png_bytes: bytes, stats_text: str) -> bytes:
        if not png_bytes:
            return b""
        try:
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except IOError:
                font = ImageFont.load_default()
                
            # Add a semi-transparent black rectangle at the bottom
            width, height = img.size
            box_height = 60
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(((0, height - box_height), (width, height)), fill=(0, 0, 0, 180))
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            
            # Draw text
            draw = ImageDraw.Draw(img)
            draw.text((10, height - box_height + 10), stats_text, font=font, fill=(255, 255, 255))
            
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Failed to create death postcard: {e}")
            return png_bytes

    # ── Command bot ───────────────────────────────────────────────────────────

    def register_command(self, command: str, handler: Callable, description: str = "") -> None:
        """Register a callable to be invoked when /command is received. Handler takes (text: str) or ()"""
        self._handlers[f"/{command}"] = handler
        if description:
            self._commands_desc.append({"command": command, "description": description})

    def register_stats_provider(self, fn: Callable[[], str]) -> None:
        """fn() should return a formatted string of current training stats."""
        self._stats_fn = fn

    @staticmethod
    def _find_noita_bounds() -> Optional[dict]:
        """Return window bounds for any window with 'noita' in title, or None."""
        try:
            import pygetwindow as gw
            matches = [t for t in gw.getAllTitles() if "noita" in t.lower() and t.strip()]
            for title in matches:
                try:
                    wins = gw.getWindowsWithTitle(title)
                    if wins:
                        w = wins[0]
                        if w.width >= 50 and w.height >= 50:
                            return {"top": max(0, w.top), "left": max(0, w.left),
                                    "width": w.width, "height": w.height}
                except Exception:
                    continue
        except Exception:
            pass
        try:
            import ctypes
            import ctypes.wintypes as wt
            found: list = []

            @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
            def _cb(hwnd, _):
                if not ctypes.windll.user32.IsWindowVisible(hwnd):
                    return True
                ln = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                if not ln:
                    return True
                buf = ctypes.create_unicode_buffer(ln + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, ln + 1)
                if "noita" in buf.value.lower():
                    r = wt.RECT()
                    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(r))
                    w, h = r.right - r.left, r.bottom - r.top
                    if w >= 50 and h >= 50:
                        found.append({"top": max(0, r.top), "left": max(0, r.left),
                                      "width": w, "height": h})
                        return False
                return True

            ctypes.windll.user32.EnumWindows(_cb, 0)
            if found:
                return found[0]
        except Exception:
            pass
        return None

    def capture_noita_screen(self, overlay_text: str = "") -> bytes:
        import io
        try:
            import mss
            from PIL import Image

            bounds = self._find_noita_bounds()
            if not bounds:
                return b""

            with mss.mss() as sct:
                monitor = bounds
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                
                if overlay_text:
                    draw = ImageDraw.Draw(img)
                    try:
                        font = ImageFont.truetype("arial.ttf", 24)
                    except IOError:
                        font = ImageFont.load_default()
                    
                    # Draw background box for readability
                    text_bbox = draw.textbbox((10, 10), overlay_text, font=font)
                    draw.rectangle([text_bbox[0]-5, text_bbox[1]-5, text_bbox[2]+5, text_bbox[3]+5], fill=(0, 0, 0, 150))
                    draw.text((10, 10), overlay_text, fill=(0, 255, 0), font=font)

                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
        except Exception as e:
            logger.error("Failed to capture screen: {}", e)
            return b""

    def _answer_callback_query(self, callback_query_id: str, text: str = ""):
        try:
            requests.post(
                f"{self._base}/answerCallbackQuery",
                json={"callback_query_id": callback_query_id, "text": text, "show_alert": False},
                timeout=5
            )
        except Exception:
            pass

    def setup_bot_menu(self) -> None:
        if not self._enabled or not self._commands_desc:
            return
        try:
            requests.post(
                f"{self._base}/setMyCommands",
                json={"commands": self._commands_desc},
                timeout=10
            )
            logger.info("Telegram bot menu commands updated")
        except Exception as e:
            logger.warning("Failed to set bot menu: {}", e)

    def start_polling(self) -> None:
        if not self._enabled:
            return
        self._running = True

        # Register built-in commands
        if "/status" not in self._handlers:
            def _status():
                text = self._stats_fn() if self._stats_fn else "No stats available yet."
                self.send_text(f"📊 <b>Status</b>\n{text}")
            self.register_command("status", _status, "Training statistics")

        if "/screen" not in self._handlers:
            def _screen():
                png = self.capture_noita_screen()
                if png:
                    self.send_photo(png, caption="📸 Capture from Noita window")
                else:
                    self.send_text("⚠️ Could not capture Noita window. Is the game running?")
            self.register_command("screen", _screen, "Capture Noita game screenshot")

        # Sync menu to Telegram
        self.setup_bot_menu()

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
                    
                    # Handle message commands
                    if "message" in update:
                        text = update["message"].get("text", "").strip()
                        
                        # Find matching command (allows arguments, e.g. /lr 0.0001)
                        for cmd_str, handler in self._handlers.items():
                            if text.startswith(cmd_str):
                                try:
                                    import inspect
                                    sig = inspect.signature(handler)
                                    if len(sig.parameters) > 0:
                                        handler(text)
                                    else:
                                        handler()
                                except Exception as exc:
                                    logger.warning("Telegram handler '{}' raised: {}", text, exc)
                                break

                    # Handle callback queries (inline buttons)
                    elif "callback_query" in update:
                        cq = update["callback_query"]
                        cq_id = cq["id"]
                        data = cq.get("data", "")

                        cmd = f"/{data}"
                        handler = self._handlers.get(cmd)
                        # Answer immediately — Telegram shows spinner until answered,
                        # and times out after 10 s. Handlers can take much longer.
                        self._answer_callback_query(cq_id)
                        if handler:
                            def _run(h=handler, d=data):
                                try:
                                    h()
                                except Exception as exc:
                                    logger.warning("Telegram callback handler '{}' raised: {}", d, exc)
                            threading.Thread(target=_run, daemon=True, name=f"tg-cb-{data}").start()
            except Exception as exc:
                logger.debug("Telegram poll error (will retry): {}", exc)
            time.sleep(2)
