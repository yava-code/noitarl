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

Actions (Discrete 10):
  0=idle  1=left  2=right  3=jump
  4=left+jump  5=right+jump  6=fire(auto-aim)  7=fire-down(dig)
  8=kick(melee)  9=jetpack_hold(ascend, burns fuel)
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Optional
from collections import Counter

import numpy as np
import websockets
import gymnasium as gym
from loguru import logger


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


class NoitaEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, host: str = "localhost", port: int = 5001):
        super().__init__()
        self.host = host
        self.port = port

        self.action_space = gym.spaces.Discrete(10)
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
        self.max_spawn_distance  = 0.0
        self.max_x               = 0.0
        self.total_damage        = 0.0
        self.episode_start_time  = time.time()
        self.steps_without_progress = 0
        self.visited_chunks: set = set()
        
        self.route_x: list[float] = []
        self.route_y: list[float] = []
        self.action_history: list[int] = []

        # Per-step event-detection state for VideoRecorder
        self._fast_mv_counter = 0        # consecutive steps with |vx|>120
        self._long_survival_triggered = False   # fires once per episode at step 600
        self._recorder = None            # injected via set_recorder()

        self._start_server()
        logger.info("[env:{}] WebSocket server on {}:{}", self.port, host, port)

    def set_recorder(self, recorder) -> None:
        """Inject VideoRecorder after construction (avoids circular import)."""
        self._recorder = recorder

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
        self.steps_without_progress   = 0
        self.visited_chunks           = set()
        self.route_x                  = []
        self.route_y                  = []
        self.action_history           = []
        self._long_survival_triggered = False

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

        # ── Reward ────────────────────────────────────────────────────────────
        # Design goal: reward ANY movement away from spawn, not just descent.
        # The old code only rewarded +Δy and punished standing still with -10,
        # which killed the agent for walking down a horizontal corridor.
        reward = -0.001  # very small time tax (was -0.005)

        # 1. Manhattan progress from spawn (rewards lateral movement too)
        dist = abs(current_x - self.spawn_x) + abs(current_y - self.spawn_y)
        if dist > self.max_spawn_distance:
            reward += (dist - self.max_spawn_distance) * 0.02
            self.max_spawn_distance = dist
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1

        # 2. Strong curiosity bonus for new 32×32 chunks — but only underground.
        # JETPACK_HOLD lets the agent fly to the ceiling; without this gate it would
        # farm chunk reward by exploring the open surface biome.
        chunk = (int(current_x // 32), int(current_y // 32))
        if chunk not in self.visited_chunks:
            self.visited_chunks.add(chunk)
            sky_vis = float(state.get("sky_visibility", 0.0))
            if sky_vis < 0.3:
                reward += 0.5

        # 3. Small additional bonus for new depth records (preserves "go down")
        if current_y > self.max_depth_y:
            reward += (current_y - self.max_depth_y) * 0.02
            self.max_depth_y = current_y

        # 4. Soft truncation when no progress for ~40s real time (no penalty —
        # truncated episodes don't bootstrap to V(s)=0 in SB3, unlike terminal).
        truncated = False
        if self.steps_without_progress > 600:
            logger.info("[env:{}] Ep truncated (no progress for 600 steps).", self.port)
            truncated = True

        # 5. Damage / kills (kept, milder)
        if current_hp < self.last_hp:
            damage = self.last_hp - current_hp
            reward -= damage * 1.0
            self.total_damage += damage

        self.max_x = max(self.max_x, current_x)

        current_kills = state.get("kills", 0)
        if current_kills > self.last_kills:
            reward += (current_kills - self.last_kills) * 5.0
        self.last_kills = current_kills

        # Chest opened by Lua auto-open (agent was close enough → EntityKill)
        current_chests = state.get("chests", 0)
        if current_chests > self.last_chests:
            reward += (current_chests - self.last_chests) * 3.0
        self.last_chests = current_chests

        # 5b. Aim-on-enemy bonus: small reward when FIRE pressed AND an enemy is in
        # 250 px. Closes the gap between "I pulled the trigger" and the rare +5/kill.
        # Capped below chunk reward so curiosity isn't overridden.
        if int(action) in (6, 7):
            enemy_radar = state.get("enemy_radar", [1.0] * 8)
            if any(v < 0.9 for v in enemy_radar):
                reward += 0.05

        # 5c. Portal proximity bonus + one-shot teleport reward (Holy Mountain).
        portal = state.get("portal", [1.0, 0.5, 0.5])
        if isinstance(portal, list) and len(portal) == 3:
            portal_dist = float(portal[0])
            if portal_dist < 0.3:
                # Smooth gradient pulling the agent into the portal.
                reward += (0.3 - portal_dist) * 0.05
        reward += portal_teleport_reward

        # 6. Death penalty — mild so the agent doesn't become risk-averse
        if dead and current_hp <= 0:
            reward -= 1.0

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

            # 6. Wand kill in this step
            if current_kills > self.last_kills and int(action) in (6, 7):
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
        
        if len(self.action_history) >= 500:
            wa = self.action_history[-500:]
            c = Counter(wa)
            if c.most_common(1)[0][1] > 400: # 80% of 500
                action_loop = True

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
                "noita/steps_without_progress": int(self.steps_without_progress),
                "noita/death_reason":           reason,
                "noita/route_x":                self.route_x,
                "noita/route_y":                self.route_y,
                "noita/visually_stuck":         visually_stuck,
                "noita/action_loop":            action_loop,
            }
            with self._lock:
                self._state = None   # force reset() to wait for fresh state
            terminated = bool(dead) and not truncated
            return self._obs_from_state(state), reward, terminated, truncated, info

        return self._obs_from_state(state), reward, False, False, {}

    def render(self) -> None:
        pass

    def close(self) -> None:
        with self._lock:
            self._ws = None
