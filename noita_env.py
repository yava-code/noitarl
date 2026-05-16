"""
Gymnasium environment bridging Python ↔ Noita via WebSocket.

Observation (60 float32, all in [0, 1]):
  [0..15]   16 platform rays       (0=wall, 1=150 px clear)
  [16..23]   8 enemy radar sectors  (1=none, 0=enemy at player)
  [24..31]   8 liquid sensors       (0=dry, ~1=pool ahead)
  [32..39]   8 projectile radar     (1=clear, 0=bullet at player)
  [40..47]   8 gold radar sectors   (1=no gold, 0=gold at player)
  [48]       hp fraction
  [49]       vx normalised  (−200..+200 → 0..1)
  [50]       vy normalised
  [51]       on_ground
  [52]       jetpack fuel   (0=empty, 1=full)
  [53]       wand ready     (0=cooldown, 1=can fire)
  [54]       is_on_fire     (0 or 1)
  [55]       is_poisoned    (0 or 1)
  [56]       sky_visibility (1=surface, 0=deep underground)
  [57]       portal distance   (1=no portal in 400px, 0=at portal)
  [58]       portal dx_norm    (0.5=portal at same X; <0.5 left, >0.5 right)
  [59]       portal dy_norm    (0.5=portal at same Y; <0.5 above, >0.5 below)

Actions (Discrete 18):
  0=idle  1=left  2=right  3=jump  4=left+jump  5=right+jump
  6=jetpack_hold(ascend)  7=kick(melee)
  8=fire_R  9=fire_UR  10=fire_U  11=fire_UL
  12=fire_L  13=fire_DL  14=fire_D  15=fire_DR
  16=fire_auto_enemy  17=fire_smart_loot
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Optional
from collections import Counter, deque

import numpy as np
import websockets
import gymnasium as gym
from loguru import logger

# CV-branch imports — lazy at module level so headless / no-Noita-yet startup
# still works. Each env instance owns its own mss handle (mss is not thread-safe).
import cv2
import mss
import win32gui
import win32process


def _capture_noita_frame() -> "Optional[Image.Image]":
    """Capture current Noita game frame via PrintWindow — works in background."""
    try:
        from video_recorder import VideoRecorder  # lazy import, no circular dep
        hwnd = VideoRecorder._find_noita_hwnd()
        if hwnd is None:
            return None
        return VideoRecorder._print_window(hwnd)
    except Exception:
        return None


def _find_noita_hwnd_by_pid(target_pid: int) -> Optional[int]:
    """Return the visible top-level HWND owned by the given Noita process, or None."""
    found: list[int] = []

    def _cb(hwnd: int, _lparam) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
        except Exception:
            return True
        if pid != target_pid:
            return True
        title = win32gui.GetWindowText(hwnd) or ""
        # Noita's main window is titled "Noita"; skip console/helper windows.
        if "Noita" in title:
            found.append(hwnd)
        return True

    try:
        win32gui.EnumWindows(_cb, None)
    except Exception as exc:
        logger.debug("EnumWindows failed: {}", exc)
    return found[0] if found else None



def _dismiss_error_dialog(target_pid: Optional[int] = None) -> bool:
    """
    Searches for Noita error/crash dialogs and clicks the 'Always Ignore'
    button if present to prevent the training from hanging.
    """
    BM_CLICK = 0x00F5
    clicked = False

    def _enum_child_cb(child_hwnd: int, _lparam) -> bool:
        nonlocal clicked
        if not win32gui.IsWindowVisible(child_hwnd):
            return True
        child_title = win32gui.GetWindowText(child_hwnd) or ""
        child_title_lower = child_title.lower()
        if "ignore" in child_title_lower and "always" in child_title_lower:
            try:
                # SendMessage is blocking, PostMessage is async
                import win32api
                import win32con
                win32api.PostMessage(child_hwnd, BM_CLICK, 0, 0)
                logger.info("Dismissed Noita crash dialog: clicked '{}' (hwnd: {})", child_title, child_hwnd)
                clicked = True
            except Exception as e:
                logger.debug("Failed to click dialog button: {}", e)
        return True

    def _enum_windows_cb(hwnd: int, _lparam) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True

        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
        except Exception:
            return True

        if target_pid is not None and pid != target_pid:
            # If target PID is provided, only look at dialogs owned by it
            return True

        title = win32gui.GetWindowText(hwnd) or ""
        title_lower = title.lower()

        # Check if this might be an error dialog (usually they have 'Noita' or 'Error' in title,
        # but their class is often '#32770' for standard dialogs).
        class_name = win32gui.GetClassName(hwnd)
        if "noita" in title_lower or "error" in title_lower or class_name == "#32770":
            try:
                win32gui.EnumChildWindows(hwnd, _enum_child_cb, None)
            except Exception:
                pass
        return True

    try:
        win32gui.EnumWindows(_enum_windows_cb, None)
    except Exception as exc:
        logger.debug("EnumWindows failed during dialog check: {}", exc)

    return clicked

def _find_any_noita_hwnd() -> Optional[int]:
    """Fallback: any visible window with 'Noita' in the title (single-instance mode)."""
    found: list[int] = []

    def _cb(hwnd: int, _lparam) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd) or ""
        if title.strip() == "Noita":
            found.append(hwnd)
        return True

    try:
        win32gui.EnumWindows(_cb, None)
    except Exception:
        pass
    return found[0] if found else None


class NoitaEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5001,
        *,
        cv_enabled: Optional[bool] = None,
        image_size: Optional[int] = None,
        frame_stack: Optional[int] = None,
        noita_pid: Optional[int] = None,
    ):
        super().__init__()
        self.host = host
        self.port = port

        # Pull defaults from Config (env vars / .env / defaults).
        from config import Config
        _cfg = Config()
        self.cv_enabled  = bool(_cfg.cv_enabled) if cv_enabled is None else bool(cv_enabled)
        self.image_size  = int(_cfg.image_size)  if image_size  is None else int(image_size)
        self.frame_stack = int(_cfg.frame_stack) if frame_stack is None else int(frame_stack)

        # CV state — lazy hwnd discovery (Noita may not be running yet at construct time).
        self.noita_pid: Optional[int] = noita_pid
        self._hwnd: Optional[int] = None
        self._sct: Optional[mss.base.MSSBase] = None  # mss instance per env (not thread-safe to share)
        self._frame_buf: deque = deque(maxlen=self.frame_stack)

        self.action_space = gym.spaces.Discrete(18)
        # +1 channel: synthetic aim overlay (drawn from last action).
        self._image_channels_total = self.frame_stack + 1
        if self.cv_enabled:
            self.observation_space = gym.spaces.Dict({
                "image": gym.spaces.Box(
                    low=0, high=255,
                    shape=(self._image_channels_total, self.image_size, self.image_size),
                    dtype=np.uint8,
                ),
                "sensors": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(60,), dtype=np.float32,
                ),
            })
        else:
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(60,), dtype=np.float32
            )

        # WebSocket state (written by WS thread, read by main thread)
        self._ws:    Optional[Any]  = None
        self._state: Optional[dict] = None
        self._lock   = threading.Lock()   # guards _ws + _state
        self._loop:  Optional[asyncio.AbstractEventLoop] = None

        # Episode stats
        self.episode_num         = 0
        self.episode_steps       = 0
        self.episode_reward      = 0.0
        self.last_hp             = 1.0
        self.last_x              = 0.0
        self.max_depth_y         = 0.0
        self.last_gold           = 0
        self.last_kills          = 0
        self.last_chests         = 0
        self.spawn_x             = 0.0
        self.spawn_y             = 0.0
        self.max_spawn_distance   = 0.0   # logged only, not rewarded
        self.max_x                = 0.0
        self.total_damage         = 0.0
        self.episode_start_time   = time.time()
        self.steps_without_descent = 0   # resets only on new depth record
        self.visited_chunks: set  = set()

        self.route_x: list[float] = []
        self.route_y: list[float] = []
        self.action_history: list[int] = []
        self._ep_breakdown: dict = {}    # cumulative reward breakdown for current episode

        # Post-portal grace periods (Holy Mountain navigation)
        self._post_portal_steps: int = 0  # suppress entry-portal proximity reward
        self._hm_grace_steps: int   = 0   # suppress descent-based truncation in HM

        # Per-step event-detection state for VideoRecorder
        self._fast_mv_counter = 0        # consecutive steps with |vx|>120
        self._long_survival_triggered = False   # fires once per episode at step 600
        self._recorder = None            # injected via set_recorder()
        self._extra_to_send: Optional[dict] = None

        self._start_server()
        logger.info("[env:{}] WebSocket server on {}:{}", self.port, host, port)

    def set_recorder(self, recorder) -> None:
        """Inject VideoRecorder after construction (avoids circular import)."""
        self._recorder = recorder

    def set_extra(self, extra: dict) -> None:
        """Set extra data (probs, saliency) to be sent with the next action."""
        self._extra_to_send = extra

    # ── WebSocket server ──────────────────────────────────────────────────────

    def _start_server(self) -> None:
        ready = threading.Event()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            async def _serve() -> None:
                async with websockets.serve(self._handle, self.host, self.port):
                    ready.set()
                    await asyncio.Future()   # run forever

            loop.run_until_complete(_serve())

        t = threading.Thread(target=_run, daemon=True, name=f"ws-{self.port}")
        t.start()
        ready.wait(timeout=5)

    async def _handle(self, ws) -> None:
        addr = ws.remote_address
        with self._lock:
            if self._ws is not None:
                # Two Noita instances connected to the same port — almost certainly
                # a misconfigured port.txt in one of the copies.
                logger.error(
                    "[env:{}] REJECTED connection from {} — port already in use! "
                    "Check that each Noita copy has a unique port.txt value.",
                    self.port, addr,
                )
                await ws.close()
                return
            self._ws = ws
        logger.info("[env:{}] Noita connected from {}", self.port, addr)
        try:
            async for raw in ws:
                try:
                    state = json.loads(raw)
                    with self._lock:
                        self._state = state
                except json.JSONDecodeError as exc:
                    logger.warning("[env:{}] Bad JSON from Noita: {}", self.port, exc)
        except websockets.exceptions.ConnectionClosed as exc:
            logger.warning("[env:{}] Noita disconnected: {}", self.port, exc)
        finally:
            with self._lock:
                if self._ws is ws:
                    self._ws    = None
                    self._state = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _send_action(self, action: int, extra: Optional[dict] = None) -> None:
        with self._lock:
            ws   = self._ws
            loop = self._loop
        if ws is None or loop is None:
            return
        try:
            payload = {"action": int(action)}
            if extra:
                payload.update(extra)
            
            asyncio.run_coroutine_threadsafe(
                ws.send(json.dumps(payload)), loop
            ).result(timeout=0.2)
        except Exception as exc:
            logger.debug("[env:{}] send_action failed: {}", self.port, exc)

    def _get_state(self) -> Optional[dict]:
        with self._lock:
            return self._state

    def _wait_for_live_state(self, timeout: float = 30.0) -> bool:
        """Block until a non-dead state arrives from Noita, or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            s = self._get_state()
            if s is not None and not s.get("dead", False):
                return True
            time.sleep(0.1)
        return False

    # ── CV branch ─────────────────────────────────────────────────────────────

    def _ensure_capture(self) -> bool:
        """Lazily acquire HWND for this env's Noita instance + mss handle."""
        if self._sct is None:
            try:
                self._sct = mss.mss()
            except Exception as exc:
                logger.debug("[env:{}] mss init failed: {}", self.port, exc)
                return False
        if self._hwnd is None:
            if self.noita_pid is not None:
                self._hwnd = _find_noita_hwnd_by_pid(self.noita_pid)
            if self._hwnd is None:
                # Fallback: pick any "Noita" window. In multi-env this will collide
                # between instances — pass noita_pid to disambiguate.
                self._hwnd = _find_any_noita_hwnd()
        return self._hwnd is not None

    # World-space crop around player (radius in world units, 1:1 with screen
    # pixels at Noita's default zoom). 160 ⇒ a 320×320 world-unit square,
    # roughly 20 player heights — tight enough to expose materials and enemies,
    # wide enough to include the closest projectile / wall the agent can hit.
    CROP_HALF_WORLD = 160.0

    def _grab_frame(self, state: Optional[dict]) -> np.ndarray:
        """Grab the Noita client area, crop a square around the player, return
        a (H, W) uint8 grayscale image."""
        size = self.image_size
        blank = np.zeros((size, size), dtype=np.uint8)
        if not self._ensure_capture():
            return blank
        try:
            # Skip captures while the window is minimised — GetClientRect
            # returns zero size, which would crash mss.
            left, top, right, bottom = win32gui.GetClientRect(self._hwnd)
            cw, ch = right - left, bottom - top
            if cw < 8 or ch < 8:
                return blank
            sx, sy = win32gui.ClientToScreen(self._hwnd, (0, 0))
            shot = self._sct.grab({"left": sx, "top": sy, "width": cw, "height": ch})
            arr = np.asarray(shot, dtype=np.uint8)  # BGRA, shape (ch, cw, 4)
            gray = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)

            # Determine the world→pixel scale from camera bounds. Falls back to
            # whole-window resize when we have no camera info yet (first reset).
            cam_w = float(state.get("cam_w", 0.0)) if state else 0.0
            cam_h = float(state.get("cam_h", 0.0)) if state else 0.0
            px    = float(state.get("x", 0.0))     if state else 0.0
            py    = float(state.get("y", 0.0))     if state else 0.0
            cam_x = float(state.get("cam_x", px))  if state else 0.0
            cam_y = float(state.get("cam_y", py))  if state else 0.0
            if cam_w > 8 and cam_h > 8:
                px_per_unit_x = cw / cam_w
                px_per_unit_y = ch / cam_h
                # Player pixel in the captured frame (camera centre maps to
                # frame centre in Noita).
                ppx = int(round((px - cam_x) * px_per_unit_x + cw / 2))
                ppy = int(round((py - cam_y) * px_per_unit_y + ch / 2))
                half_x = int(round(self.CROP_HALF_WORLD * px_per_unit_x))
                half_y = int(round(self.CROP_HALF_WORLD * px_per_unit_y))
                # Clamp the crop window inside the captured frame; if the
                # player is at the edge we shift the window in rather than
                # zero-pad (keeps the CNN seeing a consistent scale).
                x0 = max(0, min(cw - 2 * half_x, ppx - half_x))
                y0 = max(0, min(ch - 2 * half_y, ppy - half_y))
                x1 = min(cw, x0 + 2 * half_x)
                y1 = min(ch, y0 + 2 * half_y)
                crop = gray[y0:y1, x0:x1]
                if crop.size == 0:
                    crop = gray
            else:
                crop = gray
            return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        except Exception as exc:
            # HWND may have died (Noita restart). Force re-discovery next time.
            logger.debug("[env:{}] _grab_frame failed, dropping hwnd: {}", self.port, exc)
            self._hwnd = None
            return blank

    # Mapping fire action → unit aim vector (dx, dy) in screen coords (y-down).
    _FIRE_DIRS = {
        8:  ( 1.0,  0.0),   # fire_R
        9:  ( 0.7071, -0.7071),  # fire_UR
        10: ( 0.0, -1.0),   # fire_U
        11: (-0.7071, -0.7071),  # fire_UL
        12: (-1.0,  0.0),   # fire_L
        13: (-0.7071,  0.7071),  # fire_DL
        14: ( 0.0,  1.0),   # fire_D
        15: ( 0.7071,  0.7071),  # fire_DR
    }

    def _aim_channel(self, state: Optional[dict]) -> np.ndarray:
        """Synthetic image channel: white line from centre toward the agent's
        current aim direction. Empty for non-fire actions, which lets the CNN
        learn 'no line' = 'just moving / not shooting'."""
        size = self.image_size
        arr = np.zeros((size, size), dtype=np.uint8)
        if not self.action_history:
            return arr
        last_act = int(self.action_history[-1])

        direction: Optional[tuple[float, float]] = None
        if last_act in self._FIRE_DIRS:
            direction = self._FIRE_DIRS[last_act]
        elif last_act == 16 and state is not None:
            # Auto-aim: enemy_radar is 8 sectors starting at +x going CCW
            # (sector i covers angle = -i*pi/4 in screen coords).
            er = state.get("enemy_radar", [1.0] * 8)
            if len(er) == 8:
                mi = int(min(range(8), key=lambda i: er[i]))
                if er[mi] < 0.9:
                    ang = -mi * (np.pi / 4.0)
                    direction = (float(np.cos(ang)), float(np.sin(ang)))
        elif last_act == 17 and state is not None:
            gr = state.get("gold_radar", [1.0] * 8)
            if len(gr) == 8:
                mi = int(min(range(8), key=lambda i: gr[i]))
                if gr[mi] < 0.9:
                    ang = -mi * (np.pi / 4.0)
                    direction = (float(np.cos(ang)), float(np.sin(ang)))

        if direction is None:
            return arr

        dx, dy = direction
        cx = cy = size // 2
        radius = size // 2 - 2
        ex = int(round(cx + dx * radius))
        ey = int(round(cy + dy * radius))
        cv2.line(arr, (cx, cy), (ex, ey), 255, thickness=2)
        cv2.circle(arr, (cx, cy), 3, 200, -1)
        return arr

    def _push_frame(self, state: Optional[dict]) -> np.ndarray:
        """Capture a new frame, push onto the stack, append the synthetic aim
        channel, return stacked (frame_stack+1, H, W) uint8."""
        frame = self._grab_frame(state)
        if not self._frame_buf:
            # Cold start: fill stack with the same frame so CNN sees no
            # temporal noise on episode boundary.
            for _ in range(self.frame_stack):
                self._frame_buf.append(frame)
        else:
            self._frame_buf.append(frame)
        aim = self._aim_channel(state)
        stacked = np.stack(list(self._frame_buf) + [aim], axis=0)
        return stacked  # (frame_stack+1, H, W)

    def _make_obs(self, state: Optional[dict]) -> Any:
        """Build the final observation (Dict if cv_enabled, else legacy Box(60,))."""
        sensors = self._sensors_from_state(state)
        if not self.cv_enabled:
            return sensors
        return {"image": self._push_frame(state), "sensors": sensors}

    # Backwards-compat: old tests + callers expect a flat sensor vector here.
    # The full Dict observation lives in _make_obs.
    def _obs_from_state(self, state: Optional[dict]) -> np.ndarray:
        return self._sensors_from_state(state)

    def _sensors_from_state(self, state: Optional[dict]) -> np.ndarray:
        # See module docstring for the full 60-feature layout.
        if state is None:
            return np.zeros(60, dtype=np.float32)

        vx   = float(np.clip(state.get("vx", 0.0) / 200.0, -1.0, 1.0)) * 0.5 + 0.5
        vy   = float(np.clip(state.get("vy", 0.0) / 200.0, -1.0, 1.0)) * 0.5 + 0.5
        gnd  = 1.0 if state.get("on_ground", False) else 0.0
        fuel = float(state.get("jetpack_fuel", 1.0))
        wand = float(state.get("wand_ready",   1.0))

        def _padn(key, default, n):
            v = state.get(key, [default] * n)
            return (v + [default] * n)[:n] if len(v) != n else v

        rays    = _padn("rays",             1.0, 16)
        enemies = _padn("enemy_radar",      1.0, 8)
        liquids = _padn("liquid_sensors",   0.0, 8)
        projs   = _padn("projectile_radar", 1.0, 8)
        gold    = _padn("gold_radar",       1.0, 8)

        is_on_fire   = float(state.get("is_on_fire",    0.0))
        is_poisoned  = float(state.get("is_poisoned",   0.0))
        sky_vis      = float(state.get("sky_visibility", 0.0))

        portal = state.get("portal", [1.0, 0.5, 0.5])
        if not isinstance(portal, list) or len(portal) != 3:
            portal = [1.0, 0.5, 0.5]

        return np.array(
            rays + enemies + liquids + projs + gold +
            [state.get("hp", 1.0), vx, vy, gnd, fuel, wand,
             is_on_fire, is_poisoned, sky_vis,
             float(portal[0]), float(portal[1]), float(portal[2])],
            dtype=np.float32,
        )

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_num            += 1
        self.episode_steps           = 0
        self.episode_reward          = 0.0
        self.last_hp                 = 1.0
        self.max_depth_y             = 0.0
        self.last_gold               = 0
        self.last_kills              = 0
        self.last_chests             = 0
        self.max_spawn_distance      = 0.0
        self.max_x                   = 0.0
        self.total_damage            = 0.0
        self.episode_start_time      = time.time()
        self.steps_without_descent    = 0
        self.visited_chunks           = set()
        self.route_x                  = []
        self.route_y                  = []
        self.action_history           = []
        self._ep_breakdown            = {}
        self._post_portal_steps       = 0
        self._hm_grace_steps          = 0
        self._fast_mv_counter         = 0
        self._long_survival_triggered = False
        self._frame_buf.clear()        # CV: drop stale frames between episodes

        logger.debug("[env:{}] reset() — episode {}", self.port, self.episode_num)

        if not self._wait_for_live_state(timeout=60.0):
            logger.error("[env:{}] reset() timed out — is Noita running?", self.port)

        s = self._get_state()
        if s:
            self.last_hp     = s.get("hp", 1.0)
            self.max_depth_y = s.get("y", 0.0)
            self.last_x      = s.get("x", 0.0)
            self.spawn_x     = s.get("x", 0.0)
            self.spawn_y     = s.get("y", 0.0)
            self.max_x       = self.spawn_x
            self.last_gold   = s.get("gold",   0)
            self.last_kills  = s.get("kills",  0)
            self.last_chests = s.get("chests", 0)

        return self._make_obs(s), {}

    def _wait_for_new_frame(self, prev_frame: int, timeout: float = 2.0) -> Optional[dict]:
        """Block until Noita sends a state with a different frame number."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            s = self._get_state()
            if s is not None and s.get("frame", -1) != prev_frame:
                return s
            time.sleep(0.008)   # poll every 8 ms (~2x per Noita frame at 60fps)

        # timeout — return whatever we have (Noita may be loading/paused/crashed)
        _dismiss_error_dialog(self.noita_pid)
        return self._get_state()

    def step(self, action: int):
        prev_state = self._get_state()
        prev_frame = prev_state.get("frame", -1) if prev_state else -1

        self._send_action(int(action), extra=self._extra_to_send)
        self._extra_to_send = None

        # Wait for a genuinely new game frame, not stale data
        state = self._wait_for_new_frame(prev_frame)

        if state is None:
            logger.debug("[env:{}] step() with no state (disconnected?)", self.port)
            return self._make_obs(None), 0.0, False, False, {}

        current_x  = state.get("x",  0.0)
        current_y  = state.get("y",  0.0)
        current_hp = state.get("hp", 0.0)
        dead       = state.get("dead", False)


        # Portal teleport detection — sudden large Δposition between frames is a
        # holy-mountain teleporter trigger (real walking caps at ~60 px/step).
        # Guarded by "was near a portal last frame" so a WebSocket reconnect or
        # Noita restart doesn't trigger a false +20.
        portal_teleport_reward = 0.0
        if prev_state is not None and not prev_state.get("dead", False):
            prev_x = prev_state.get("x", current_x)
            prev_y = prev_state.get("y", current_y)
            prev_portal = prev_state.get("portal", [1.0, 0.5, 0.5])
            was_near_portal = (
                isinstance(prev_portal, list) and len(prev_portal) == 3
                and float(prev_portal[0]) < 0.5
            )
            big_jump = abs(current_x - prev_x) > 300 or abs(current_y - prev_y) > 300
            if big_jump and was_near_portal:
                portal_teleport_reward = 20.0
                logger.info(
                    "[env:{}] Portal teleport detected (Δ=({:.0f},{:.0f})) → +20",
                    self.port, current_x - prev_x, current_y - prev_y,
                )

        # ── Post-teleport spawn reset (Holy Mountain) ─────────────────────────
        # After entering a Holy Mountain portal, the agent lands in a new area
        # where previous depth/distance records are irrelevant. Reset the origin
        # so Manhattan + depth rewards fire fresh from the new location.
        if portal_teleport_reward > 0:
            self.spawn_x            = current_x
            self.spawn_y            = current_y
            self.max_spawn_distance = 0.0
            self.max_depth_y        = current_y
            self.visited_chunks     = set()
            self.steps_without_descent = 0
            # Suppress entry-portal proximity pull for ~13 s (200 steps) so the
            # agent explores HM rightward instead of turning back through entry portal.
            self._post_portal_steps = 200
            # Allow HM horizontal exploration without descent-based truncation.
            self._hm_grace_steps    = 800
            logger.info(
                "[env:{}] Post-portal spawn reset → ({:.0f}, {:.0f})",
                self.port, current_x, current_y,
            )

        # ── Reward ────────────────────────────────────────────────────────────
        r_time   = -0.001  # small time tax

        # 1. Manhattan progress: rewards any movement that expands the frontier,
        #    giving the agent a dense signal while it learns the topology.
        r_manh = 0.0
        dist = abs(current_x - self.spawn_x) + abs(current_y - self.spawn_y)
        if dist > self.max_spawn_distance:
            r_manh = (dist - self.max_spawn_distance) * 0.015
            self.max_spawn_distance = dist

        # 2. Depth — weighted 2× higher than Manhattan so the agent prefers
        #    going DOWN over going sideways at the same speed.
        r_depth = 0.0
        if current_y > self.max_depth_y:
            r_depth = (current_y - self.max_depth_y) * 0.03
            self.max_depth_y = current_y
            self.steps_without_descent = 0
        else:
            self.steps_without_descent += 1

        # 2b. Dense downward bias — fires every step the agent moves deeper.
        # Unlike r_depth (only on new record), this gives a gradient in corridors
        # where the agent needs to go sideways before it can descend.
        r_down_bias = 0.0
        if prev_state is not None and not prev_state.get("dead", False):
            _prev_y = prev_state.get("y", current_y)
            _delta_y = current_y - _prev_y
            if _delta_y > 0:
                r_down_bias = _delta_y * 0.005

        # 3. Chunk curiosity — underground only (sky-farm guard).
        # New chunk also resets the descent timer: horizontal navigation through
        # corridors is progress, not stagnation.  Without this, the agent learns
        # that any horizontal step is "bad" (timer runs toward TRUNC -2).
        r_chunk = 0.0
        chunk = (int(current_x // 32), int(current_y // 32))
        if chunk not in self.visited_chunks:
            self.visited_chunks.add(chunk)
            sky_vis = float(state.get("sky_visibility", 0.0))
            if sky_vis < 0.3:
                r_chunk = 0.5
                self.steps_without_descent = 0   # exploration = progress

        # 4. Truncation: no depth progress for 500 steps (~33 s).
        # After a portal teleport, grant HM grace period so the agent can
        # navigate the horizontal Holy Mountain without being truncated for
        # "no descent" — HM is lateral, not vertical.
        truncated = False
        r_trunc = 0.0
        if self._hm_grace_steps > 0:
            self._hm_grace_steps -= 1
        elif self.steps_without_descent > 500:
            r_trunc = -2.0
            truncated = True
            self._send_action(-1)
            logger.info(
                "[env:{}] Truncated — no descent for 500 steps (-2 penalty).",
                self.port,
            )

        # 5. Damage / kills
        r_dmg = 0.0
        if current_hp < self.last_hp:
            damage = self.last_hp - current_hp
            r_dmg = -damage * 1.0
            self.total_damage += damage

        self.max_x = max(self.max_x, current_x)

        r_kills = 0.0
        current_kills = state.get("kills", 0)
        if current_kills > self.last_kills:
            r_kills = (current_kills - self.last_kills) * 5.0
        self.last_kills = current_kills

        r_chests = 0.0
        current_chests = state.get("chests", 0)
        if current_chests > self.last_chests:
            r_chests = (current_chests - self.last_chests) * 3.0
        self.last_chests = current_chests

        # 5b. Aim-on-enemy bonus: small reward when wand FIRED AND enemy in 250 px.
        # Fire actions: 8=fire_R … 16=fire_auto_enemy (excludes 17=fire_smart_loot).
        # Bumped 0.01 → 0.03 so it survives next to chunk (+0.5) / depth signals.
        r_fire  = 0.0
        # 5b-bis. Wasted-shot penalty: agent fired with no enemy in range OR
        # while the wand was on cooldown. Discourages spam-firing into walls
        # and the "stuck-in-fire-loop" failure mode.
        r_waste = 0.0
        if int(action) in range(8, 17):   # 8..16 = all directional + auto-aim fire
            enemy_radar = state.get("enemy_radar", [1.0] * 8)
            wand_ready  = float(state.get("wand_ready", 1.0))
            enemy_visible = any(v < 0.9 for v in enemy_radar)
            if enemy_visible and wand_ready >= 0.5:
                r_fire = 0.03
            else:
                # No legitimate reason to be pulling the trigger right now.
                r_waste = -0.05

        # 5d. Terrain destruction bonus: wand fire visibly extended a ray = terrain broke.
        # Maps each directional fire action to the ray index it fires along.
        # Only rewards when ray actually got *meaningfully* longer (>45 px of
        # the 150 px range, i.e. >0.30 normalised) — small flickers from sand
        # falling no longer count.
        r_dig = 0.0
        _FIRE_TO_RAY = {8: 0, 9: 14, 10: 12, 11: 10, 12: 8, 13: 6, 14: 4, 15: 2}
        if (int(action) in _FIRE_TO_RAY and
                prev_state is not None and not prev_state.get("dead", False)):
            _ri = _FIRE_TO_RAY[int(action)]
            _pr = prev_state.get("rays", [])
            _cr = state.get("rays", [])
            if len(_pr) > _ri and len(_cr) > _ri and _cr[_ri] - _pr[_ri] > 0.30:
                r_dig = 0.5   # ray extended >45 px: terrain was destroyed

        # 5c. Portal proximity bonus + one-shot teleport reward (Holy Mountain).
        # During _post_portal_steps: suppress ALL proximity (entry portal right behind agent).
        # During _hm_grace_steps: only reward portals that are to the RIGHT (exit portal)
        # so the entry portal (to the left) doesn't pull the agent back.
        r_portal = 0.0
        if self._post_portal_steps > 0:
            self._post_portal_steps -= 1
        else:
            portal = state.get("portal", [1.0, 0.5, 0.5])
            if isinstance(portal, list) and len(portal) == 3:
                portal_dist = float(portal[0])
                portal_dx   = float(portal[1])   # 0.5=same X, >0.5=right, <0.5=left
                in_hm = self._hm_grace_steps > 0
                if portal_dist < 0.3 and (not in_hm or portal_dx > 0.5):
                    r_portal = (0.3 - portal_dist) * 0.05
        r_portal += portal_teleport_reward

        # 6. Death penalty
        r_death = 0.0
        if dead and current_hp <= 0:
            r_death = -1.0

        reward = (r_time + r_manh + r_depth + r_down_bias + r_chunk + r_dig
                  + r_trunc + r_dmg + r_kills + r_chests + r_fire + r_waste
                  + r_portal + r_death)

        # Accumulate per-episode reward totals (sent at episode end for /plot analysis)
        for _k, _v in (("time", r_time), ("manh", r_manh), ("depth", r_depth),
                       ("down", r_down_bias), ("chunk", r_chunk), ("dig", r_dig),
                       ("fire", r_fire), ("waste", r_waste), ("portal", r_portal),
                       ("kills", r_kills), ("dmg", r_dmg), ("death", r_death), ("trunc", r_trunc)):
            self._ep_breakdown[_k] = self._ep_breakdown.get(_k, 0.0) + _v

        # Debug: log reward breakdown every 50 steps so we can see what's driving the policy
        if self.episode_steps % 50 == 0:
            logger.debug(
                "[env:{}] reward breakdown: time={:+.3f} manh={:+.3f} depth={:+.3f}"
                " down={:+.3f} chunk={:+.2f} dig={:+.2f} fire={:+.2f} waste={:+.2f}"
                " portal={:+.3f} kills={:+.1f} dmg={:+.2f} death={:+.1f} total={:+.4f}",
                self.port, r_time, r_manh, r_depth, r_down_bias, r_chunk, r_dig,
                r_fire, r_waste, r_portal, r_kills, r_dmg, r_death, reward,
            )

        self.last_hp         = current_hp
        self.episode_steps  += 1
        self.episode_reward += reward

        self.route_x.append(current_x)
        self.route_y.append(current_y)
        self.action_history.append(int(action))

        # ── Per-step VideoRecorder triggers ──────────────────────────────────
        if self._recorder is not None:
            rec = self._recorder
            ctx_base = {
                "dist":    self.max_spawn_distance,
                "depth":   self.max_depth_y,
                "kills":   current_kills,
                "steps":   self.episode_steps,
                "reward":  reward,
                "episode": self.episode_num,
                "chunks":  len(self.visited_chunks),
            }

            # 1. Portal teleport
            if portal_teleport_reward > 0:
                rec.trigger_event("portal_teleport", ctx_base)

            # 2. Reward spike (portal or big chunk burst)
            if reward > 15.0:
                rec.trigger_event("reward_spike", {**ctx_base, "reward": reward})

            # 3. Instant massive damage (HP drops ≥ 40% in one step)
            if prev_state is not None and not dead:
                prev_hp = prev_state.get("hp", 1.0)
                hp_drop = prev_hp - current_hp
                if hp_drop >= 0.40:
                    rec.trigger_event("instant_damage", {**ctx_base, "damage": hp_drop})

            # 4. Extreme fall (vy > 350 — free-falling into a pit)
            current_vy = state.get("vy", 0.0)
            if current_vy > 350:
                rec.trigger_event("extreme_fall", {**ctx_base, "vy": current_vy})

            # 5. Sustained fast horizontal movement (|vx| > 120 for 15 steps)
            current_vx_raw = state.get("vx", 0.0)
            if abs(current_vx_raw) > 120:
                self._fast_mv_counter += 1
                if self._fast_mv_counter == 15:   # trigger once per burst
                    rec.trigger_event("fast_movement", {**ctx_base, "vx": current_vx_raw})
            else:
                self._fast_mv_counter = 0

            # 6. Wand kill in this step (fire actions 8-17, not KICK/JETPACK)
            if current_kills > self.last_kills and int(action) in range(8, 18):
                rec.trigger_event("wand_kill", {**ctx_base, "kills": current_kills})

            # 7. Long survival milestone — fires DURING the episode (at step 600)
            # so the VideoRecorder pre-buffer contains actual ongoing gameplay,
            # not the respawn screen that would appear if we triggered post-episode.
            if self.episode_steps == 600 and not self._long_survival_triggered:
                self._long_survival_triggered = True
                rec.trigger_event("long_survival", {**ctx_base,
                    "steps": self.episode_steps, "reward": self.episode_reward})

        visually_stuck = False
        action_loop = False
        if len(self.route_x) >= 200:
            wx = self.route_x[-200:]
            wy = self.route_y[-200:]
            if max(wx) - min(wx) < 20 and max(wy) - min(wy) < 20:
                visually_stuck = True

        # Only IDLE (0) counts as a loop: standing still doing nothing.
        # Movement (1-5), JETPACK (6), KICK (7), and fire (8-17) are all
        # legitimate actions that should not trigger truncation.
        _FARMABLE = {0}
        if len(self.action_history) >= 500:
            wa = self.action_history[-500:]
            top_action, top_count = Counter(wa).most_common(1)[0]
            if top_count > 400 and top_action in _FARMABLE:  # 80% IDLE
                action_loop = True

        # Action-loop truncation: degenerate policy (e.g. fire/idle farming).
        if action_loop and not truncated:
            reward -= 3.0
            truncated = True
            self._send_action(-1)
            logger.info(
                "[env:{}] Truncated — action loop (action {} >80%% of 500 steps, -3 penalty).",
                self.port, self.action_history[-1],
            )

        if dead or truncated:
            reason = "TRUNC" if truncated and not dead else "DEAD"
            logger.info(
                "[env:{}] Ep {:3d} done — steps={} reward={:.2f} "
                "max_dist={:.0f} max_depth={:.0f} chunks={} ({})",
                self.port, self.episode_num, self.episode_steps,
                self.episode_reward, self.max_spawn_distance,
                self.max_depth_y, len(self.visited_chunks),
                reason,
            )
            
            run_time = time.time() - self.episode_start_time
            info = {
                "episode": {"r": self.episode_reward, "l": self.episode_steps},
                "noita/visited_chunks":         len(self.visited_chunks),
                "noita/max_spawn_distance":     float(self.max_spawn_distance),
                "noita/max_depth":              float(self.max_depth_y),
                "noita/max_x":                  float(self.max_x),
                "noita/kills":                  int(self.last_kills),
                "noita/chests_opened":          int(self.last_chests),
                "noita/total_damage":           float(self.total_damage),
                "noita/run_time_s":             float(run_time),
                "noita/steps_without_descent":  int(self.steps_without_descent),
                "noita/death_reason":           reason,
                "noita/route_x":                self.route_x,
                "noita/route_y":                self.route_y,
                "noita/visually_stuck":         visually_stuck,
                "noita/action_loop":            action_loop,
                "noita/reward_breakdown": dict(self._ep_breakdown),  # cumulative episode totals
                "noita/action_history": self.action_history,
            }
            with self._lock:
                self._state = None   # force reset() to wait for fresh state
            terminated = bool(dead) and not truncated
            return self._make_obs(state), reward, terminated, truncated, info

        return self._make_obs(state), reward, False, False, {}

    def render(self) -> None:
        pass

    def close(self) -> None:
        with self._lock:
            self._ws = None
        # Release the mss screen-grab handle (frees DC + GDI objects).
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
        self._hwnd = None
        self._frame_buf.clear()
