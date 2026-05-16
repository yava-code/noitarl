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
            {"text": "📊 Status",  "callback_data": "status"},
            {"text": "📈 Plot",    "callback_data": "plot"},
            {"text": "📸 Screen",  "callback_data": "screen"},
            {"text": "🎥 Record",  "callback_data": "record"},
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
            import json as _json
            data = {"chat_id": self._chat_id, "caption": caption, "parse_mode": "HTML"}
            if reply_markup is None:
                data["reply_markup"] = _json.dumps({
                    "inline_keyboard": [self._buttons_row_1, self._buttons_row_2, self._buttons_row_3]
                })
            elif reply_markup:
                data["reply_markup"] = _json.dumps(reply_markup)

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
            import json as _json
            data = {"chat_id": self._chat_id, "caption": caption}
            if reply_markup is None:
                data["reply_markup"] = _json.dumps({
                    "inline_keyboard": [self._buttons_row_1, self._buttons_row_2, self._buttons_row_3]
                })
            elif reply_markup:
                data["reply_markup"] = _json.dumps(reply_markup)

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

    @staticmethod
    def make_rich_collage(
        rewards: list[float],
        route_x: list[float],
        route_y: list[float],
        actions: list[int],
        reward_breakdown: dict,
        title: str = "NoitaRL Performance Overview"
    ) -> bytes:
        """Create a 2x2 collage of charts for a deep dive into the agent's brains."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Reward Curve (Top Left)
        ax1 = axes[0, 0]
        ax1.plot(rewards, color='blue', alpha=0.3)
        if len(rewards) > 10:
            import pandas as pd
            sma = pd.Series(rewards).rolling(10).mean()
            ax1.plot(sma, color='red', linewidth=2, label='SMA 10')
        ax1.set_title("Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.grid(True, alpha=0.3)
        
        # 2. GPS Track (Top Right)
        ax2 = axes[0, 1]
        if route_x and route_y:
            ax2.plot(route_x, route_y, color='green', alpha=0.7)
            ax2.scatter(route_x[0], route_y[0], color='blue', s=50, label='Start', zorder=5)
            ax2.scatter(route_x[-1], route_y[-1], color='red', s=50, label='End', zorder=5)
            ax2.invert_yaxis() # Noita's Y is positive downwards
            ax2.set_title("Best Run GPS Track")
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "No route data", ha='center', va='center')
        ax2.set_aspect('equal', 'datalim')
        
        # 3. Action Distribution (Bottom Left)
        ax3 = axes[1, 0]
        if actions:
            from collections import Counter
            # Map action IDs to names
            ACTION_NAMES = {
                0:"IDLE", 1:"LEFT", 2:"RIGHT", 3:"JUMP",
                4:"L+JMP", 5:"R+JMP", 6:"FIRE", 7:"DIG↓",
                8:"KICK", 9:"JETPACK",
            }
            counts = Counter(actions)
            labels = [ACTION_NAMES.get(a, str(a)) for a in counts.keys()]
            ax3.pie(counts.values(), labels=labels, autopct='%1.1f%%', startangle=140)
            ax3.set_title("Action Distribution (Last Ep)")
        else:
            ax3.text(0.5, 0.5, "No action data", ha='center', va='center')
            
        # 4. Reward Breakdown (Bottom Right)
        ax4 = axes[1, 1]
        if reward_breakdown:
            # Filter zero/tiny values for clarity
            filtered = {k: v for k, v in reward_breakdown.items() if abs(v) > 0.01}
            keys = list(filtered.keys())
            vals = list(filtered.values())
            colors = ['green' if v > 0 else 'red' for v in vals]
            ax4.barh(keys, vals, color=colors)
            ax4.set_title("Reward Breakdown (Last Ep)")
            ax4.axvline(0, color='black', linewidth=0.8)
        else:
            ax4.text(0.5, 0.5, "No breakdown data", ha='center', va='center')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        return buf.getvalue()

    def generate_ai_status(self, groq_key: str, stats_context: str) -> str:
        """Call Groq to generate a smart, analytical summary of the training status."""
        if not groq_key:
            return "⚠️ Groq API key not set. Please set GROQ_API_KEY in .env."
            
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            
            prompt = (
                "You are an expert AI Analyst for a Reinforcement Learning project 'NoitaRL'.\n"
                "A PPO agent is learning to play Noita, a physics-based roguelite.\n"
                "You are provided with a 'mini-MCP' data dump including recent metrics, logs, "
                "reward breakdowns, and action histories.\n\n"
                "== PROJECT CONTEXT ==\n"
                "- Goal: Reach maximum depth and explore maximum spawn distance (Mines).\n"
                "- Rewards: Manhattan distance, depth, chunk discovery, kills, chests.\n"
                "- Penalties: Damage, death, timeout (no descent), action loops.\n\n"
                f"== CURRENT STATUS DATA ==\n{stats_context}\n\n"
                "Write a HIGHLY ANALYTICAL and CONCISE (2-3 paragraphs) summary for the user in Telegram.\n"
                "1. Identify the agent's current 'behavioral mode' (e.g., 'aggressive explorer', 'cautious looter', 'stuck in local minima').\n"
                "2. Highlight specific metrics that show progress or regression (reference numbers).\n"
                "3. Suggest what the agent might be 'thinking' or struggling with (e.g., 'it avoids projectiles well but fails to commit to depth').\n"
                "Use bold text for key terms. Use technical but readable tone. NO filler or introductions."
            )
            
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_completion_tokens=400,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Groq AI status failed: {}", e)
            return f"⚠️ AI Analysis failed: {str(e)}"

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
    def _find_noita_hwnd() -> Optional[int]:
        """
        Return HWND of the Noita game window, matched by process executable
        (noita.exe / noita_dev.exe) so browser tabs with 'noita' in their title
        are never accidentally selected.
        """
        try:
            import ctypes
            import ctypes.wintypes as wt

            k32    = ctypes.windll.kernel32
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
                hproc = k32.OpenProcess(PROCESS_QUERY_LIMITED, False, pid.value)
                if not hproc:
                    return True
                try:
                    buf  = ctypes.create_unicode_buffer(260)
                    size = ctypes.c_uint32(260)
                    if k32.QueryFullProcessImageNameW(hproc, 0, buf, ctypes.byref(size)):
                        exe = buf.value.lower().rsplit("\\", 1)[-1]
                        if exe in ("noita.exe", "noita_dev.exe"):
                            rect = wt.RECT()
                            user32.GetClientRect(hwnd, ctypes.byref(rect))
                            if rect.right >= 50 and rect.bottom >= 50:
                                found.append(int(hwnd))
                                return False
                finally:
                    k32.CloseHandle(hproc)
                return True

            user32.EnumWindows(_cb, 0)
            return found[0] if found else None
        except Exception:
            return None

    @staticmethod
    def _print_window_to_image(hwnd: int) -> Optional["Image.Image"]:
        """Render window into PIL Image via PrintWindow — works in background."""
        try:
            import ctypes
            import ctypes.wintypes as wt
            from PIL import Image

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

            # PW_CLIENTONLY(1) | PW_RENDERFULLCONTENT(2) — renders GPU/OpenGL content
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
            bih.biSize = ctypes.sizeof(_BIH)
            bih.biWidth = w
            bih.biHeight = -h
            bih.biPlanes = 1
            bih.biBitCount = 32
            bih.biCompression = 0

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
            logger.debug("notify: PrintWindow failed: {}", exc)
            return None

    def capture_noita_screen(self, overlay_text: str = "") -> bytes:
        import io
        try:
            from PIL import Image

            hwnd = self._find_noita_hwnd()
            if hwnd is None:
                return b""

            img = self._print_window_to_image(hwnd)
            if img is None:
                return b""

            if overlay_text:
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except IOError:
                    font = ImageFont.load_default()
                text_bbox = draw.textbbox((10, 10), overlay_text, font=font)
                draw.rectangle(
                    [text_bbox[0]-5, text_bbox[1]-5, text_bbox[2]+5, text_bbox[3]+5],
                    fill=(0, 0, 0, 150),
                )
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
                            def _run(h=handler, d=data, cmd_str=cmd):
                                try:
                                    import inspect
                                    sig = inspect.signature(h)
                                    if len(sig.parameters) > 0:
                                        h(cmd_str)
                                    else:
                                        h()
                                except Exception as exc:
                                    logger.warning("Telegram callback handler '{}' raised: {}", d, exc)
                            threading.Thread(target=_run, daemon=True, name=f"tg-cb-{data}").start()
            except Exception as exc:
                logger.debug("Telegram poll error (will retry): {}", exc)
            time.sleep(2)
