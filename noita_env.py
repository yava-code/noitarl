"""
Gymnasium environment bridging Python ↔ Noita via WebSocket.

Observation (57 float32, all in [0, 1]):
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

Actions (Discrete 5):  0=idle  1=left  2=right  3=jump  4=fire
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Optional

import numpy as np
import websockets
import gymnasium as gym
from loguru import logger


class NoitaEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, host: str = "localhost", port: int = 5001):
        super().__init__()
        self.host = host
        self.port = port

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(57,), dtype=np.float32
        )

        # WebSocket state (written by WS thread, read by main thread)
        self._ws:    Optional[Any]  = None
        self._state: Optional[dict] = None
        self._lock   = threading.Lock()   # guards _ws + _state
        self._loop:  Optional[asyncio.AbstractEventLoop] = None

        # Episode stats
        self.episode_num    = 0
        self.episode_steps  = 0
        self.episode_reward = 0.0
        self.last_hp        = 1.0
        self.last_x         = 0.0
        self.max_depth_y    = 0.0
        self.last_gold      = 0
        self.last_kills     = 0

        self._start_server()
        logger.info("[env:{}] WebSocket server on {}:{}", port, host, port)

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
        logger.info("[env:{}] Noita connected from {}", self.port, addr)
        with self._lock:
            self._ws = ws
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

    def _send_action(self, action: int) -> None:
        with self._lock:
            ws   = self._ws
            loop = self._loop
        if ws is None or loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                ws.send(json.dumps(action)), loop
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

    def _obs_from_state(self, state: Optional[dict]) -> np.ndarray:
        # Layout (57 total):
        #  [0..15]   platform rays      (1=clear, 0=wall)
        #  [16..23]  enemy radar        (1=none, 0=enemy at player)
        #  [24..31]  liquid sensors     (0=dry, ~1=pool ahead)
        #  [32..39]  projectile radar   (1=clear, 0=bullet at player)
        #  [40..47]  gold radar         (1=no gold, 0=gold at player)
        #  [48]      hp
        #  [49]      vx normalised
        #  [50]      vy normalised
        #  [51]      on_ground
        #  [52]      jetpack fuel
        #  [53]      wand ready
        #  [54]      is_on_fire
        #  [55]      is_poisoned
        #  [56]      sky_visibility
        if state is None:
            return np.zeros(57, dtype=np.float32)

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

        return np.array(
            rays + enemies + liquids + projs + gold +
            [state.get("hp", 1.0), vx, vy, gnd, fuel, wand,
             is_on_fire, is_poisoned, sky_vis],
            dtype=np.float32,
        )

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_num    += 1
        self.episode_steps   = 0
        self.episode_reward  = 0.0
        self.last_hp         = 1.0
        self.max_depth_y     = 0.0
        self.last_gold       = 0
        self.last_kills      = 0
        self.visited_chunks: set = set()

        logger.debug("[env:{}] reset() — episode {}", self.port, self.episode_num)

        if not self._wait_for_live_state(timeout=60.0):
            logger.error("[env:{}] reset() timed out — is Noita running?", self.port)

        s = self._get_state()
        if s:
            self.last_hp     = s.get("hp", 1.0)
            self.max_depth_y = s.get("y", 0.0)
            self.last_x      = s.get("x", 0.0)
            self.last_gold   = s.get("gold",  0)
            self.last_kills  = s.get("kills", 0)

        return self._obs_from_state(s), {}

    def _wait_for_new_frame(self, prev_frame: int, timeout: float = 2.0) -> Optional[dict]:
        """Block until Noita sends a state with a different frame number."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            s = self._get_state()
            if s is not None and s.get("frame", -1) != prev_frame:
                return s
            time.sleep(0.008)   # poll every 8 ms (~2x per Noita frame at 60fps)
        # timeout — return whatever we have (Noita may be loading/paused)
        return self._get_state()

    def step(self, action: int):
        prev_state = self._get_state()
        prev_frame = prev_state.get("frame", -1) if prev_state else -1

        self._send_action(int(action))

        # Wait for a genuinely new game frame, not stale data
        state = self._wait_for_new_frame(prev_frame)

        if state is None:
            logger.debug("[env:{}] step() with no state (disconnected?)", self.port)
            return self._obs_from_state(None), 0.0, False, False, {}

        current_x  = state.get("x",  0.0)
        current_y  = state.get("y",  0.0)
        current_hp = state.get("hp", 0.0)
        dead       = state.get("dead", False)

        # ── Reward ────────────────────────────────────────────────────────────
        # Time tax: стоять невыгодно.
        reward = -0.005

        # Curiosity: первое посещение чанка 64×64 пикселя.
        chunk = (int(current_x // 64), int(current_y // 64))
        if chunk not in self.visited_chunks:
            self.visited_chunks.add(chunk)
            reward += 0.05

        # Новая максимальная глубина.
        if current_y > self.max_depth_y:
            reward += (current_y - self.max_depth_y) * 0.5
            self.max_depth_y = current_y

        # Штраф за урон.
        if current_hp < self.last_hp:
            reward -= (self.last_hp - current_hp) * 3.0

        # Награда за сбор золота.
        current_gold = state.get("gold", 0)
        if current_gold > self.last_gold:
            reward += (current_gold - self.last_gold) * 0.001
        self.last_gold = current_gold

        # Награда за убийства.
        current_kills = state.get("kills", 0)
        if current_kills > self.last_kills:
            reward += (current_kills - self.last_kills) * 10.0
        self.last_kills = current_kills

        # Reward shaping: поощряем стрельбу по врагу, штрафуем стрельбу в пустоту.
        if action == 4:
            enemy_radar = state.get("enemy_radar", [1.0] * 8)
            if any(v < 1.0 for v in enemy_radar):
                reward += 0.05   # враг виден — молодец, стреляешь
            else:
                reward -= 0.02   # стреляешь в пустоту — штраф

        # Лимит шагов: 4000 шагов × FRAME_SKIP 4 = 16 000 кадров ≈ 4.5 мин.
        # Если бот дожил — он нашёл безопасную нору и тупит. Штраф + принудительный конец.
        if self.episode_steps >= 4000:
            dead = True
            reward -= 5.0

        # Штраф за смерть.
        if dead:
            reward -= 1.0

        self.last_hp         = current_hp
        self.episode_steps  += 1
        self.episode_reward += reward

        if dead:
            logger.info(
                "[env:{}] Ep {:3d} done — steps={} reward={:.2f} max_depth={:.0f}",
                self.port, self.episode_num, self.episode_steps,
                self.episode_reward, self.max_depth_y,
            )
            info = {"episode": {"r": self.episode_reward, "l": self.episode_steps}}
            with self._lock:
                self._state = None   # force reset() to wait for fresh state
            return self._obs_from_state(state), reward, True, False, info

        return self._obs_from_state(state), reward, False, False, {}

    def render(self) -> None:
        pass

    def close(self) -> None:
        with self._lock:
            self._ws = None
