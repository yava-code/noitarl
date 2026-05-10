"""
Gymnasium environment bridging Python ↔ Noita via WebSocket.

Observation (20 float32, all in [0, 1]):
  [0..15]  16 RaytracePlatforms sensors (0=wall at player, 1=150px clear)
  [16]     hp fraction
  [17]     vx normalised  (−200..+200 → 0..1)
  [18]     vy normalised
  [19]     on_ground

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
            low=0.0, high=1.0, shape=(25,), dtype=np.float32
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
        if state is None:
            return np.zeros(25, dtype=np.float32)
        vx  = float(np.clip(state.get("vx", 0.0) / 200.0, -1.0, 1.0)) * 0.5 + 0.5
        vy  = float(np.clip(state.get("vy", 0.0) / 200.0, -1.0, 1.0)) * 0.5 + 0.5
        gnd = 1.0 if state.get("on_ground", False) else 0.0

        rays = state.get("rays", [1.0] * 16)
        if len(rays) != 16:
            logger.warning("[env:{}] Expected 16 rays, got {}", self.port, len(rays))
            rays = (rays + [1.0] * 16)[:16]

        # 5 liquid sensors (0=dry, ~1=pool ahead); default 0 if not sent (dead state)
        liquids = state.get("liquid_sensors", [0.0] * 5)
        if len(liquids) != 5:
            liquids = (liquids + [0.0] * 5)[:5]

        return np.array(rays + [state.get("hp", 1.0), vx, vy, gnd] + liquids, dtype=np.float32)

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_num    += 1
        self.episode_steps   = 0
        self.episode_reward  = 0.0
        self.last_hp         = 1.0
        self.max_depth_y     = 0.0

        logger.debug("[env:{}] reset() — episode {}", self.port, self.episode_num)

        if not self._wait_for_live_state(timeout=60.0):
            logger.error("[env:{}] reset() timed out — is Noita running?", self.port)

        s = self._get_state()
        if s:
            self.last_hp     = s.get("hp", 1.0)
            self.max_depth_y = s.get("y", 0.0)
            self.last_x      = s.get("x", 0.0)

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
        # Time tax: стоять невыгодно, единственный способ в плюс — идти глубже.
        reward = -0.005

        # Единственный источник плюса — новая глубина.
        if current_y > self.max_depth_y:
            reward += (current_y - self.max_depth_y) * 0.5
            self.max_depth_y = current_y

        # Штраф за урон.
        if current_hp < self.last_hp:
            reward -= (self.last_hp - current_hp) * 3.0

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
