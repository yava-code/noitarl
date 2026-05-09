import gymnasium as gym
import numpy as np
import asyncio
import json
import websockets
import threading
import time


class NoitaEnv(gym.Env):
    """
    Gymnasium env wrapping Noita via WebSocket.

    Observation (20 floats, all in [0, 1]):
        [0..15]  16 raytrace sensors (0=wall touching, 1=150px clear)
        [16]     hp fraction
        [17]     vx normalised  (-200..+200 → 0..1)
        [18]     vy normalised
        [19]     on_ground (0 or 1)

    Actions (Discrete 5):
        0=idle  1=left  2=right  3=jump  4=fire
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, host="localhost", port=5001):
        super().__init__()
        self.action_space      = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(20,), dtype=np.float32
        )

        self.host = host
        self.port = port

        self.websocket     = None
        self.current_state = None   # dict written by WS thread, read by main thread
        self.loop          = None

        # Episode stats
        self.episode_num     = 0
        self.episode_steps   = 0
        self.episode_reward  = 0.0
        self.last_hp         = 1.0
        self.max_depth_y     = 0.0   # largest Y seen (Noita: +Y = deeper underground)

        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        print(f"[NoitaEnv] WebSocket server started — waiting for Noita on {host}:{port}")

    # ── WebSocket server ──────────────────────────────────────────────────

    def _run_server(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        async def _main():
            async with websockets.serve(self._handle, self.host, self.port):
                await asyncio.Future()
        self.loop.run_until_complete(_main())

    async def _handle(self, websocket):
        print("[NoitaEnv] Noita connected!")
        self.websocket = websocket
        try:
            async for raw in websocket:
                self.current_state = json.loads(raw)
        except websockets.exceptions.ConnectionClosed:
            print("[NoitaEnv] Noita disconnected.")
            self.websocket     = None
            self.current_state = None

    # ── Observation ───────────────────────────────────────────────────────

    def _get_obs(self):
        s = self.current_state
        if s is None:
            return np.zeros(20, dtype=np.float32)
        vx  = float(np.clip(s.get("vx", 0) / 200.0, -1, 1)) * 0.5 + 0.5
        vy  = float(np.clip(s.get("vy", 0) / 200.0, -1, 1)) * 0.5 + 0.5
        gnd = 1.0 if s.get("on_ground", False) else 0.0
        return np.array(s["rays"] + [s["hp"], vx, vy, gnd], dtype=np.float32)

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_num    += 1
        self.episode_steps   = 0
        self.episode_reward  = 0.0
        self.last_hp         = 1.0
        self.max_depth_y     = 0.0

        print(f"[NoitaEnv] reset() — episode {self.episode_num}, waiting for live state...")
        # Lua already respawned the player; wait for a non-dead state
        deadline = time.time() + 30.0
        while True:
            s = self.current_state
            if s is not None and not s.get("dead", False):
                break
            if time.time() > deadline:
                print("[NoitaEnv] WARNING: reset() timed out — is Noita running?")
                break
            time.sleep(0.1)

        if self.current_state:
            self.last_hp     = self.current_state.get("hp", 1.0)
            self.max_depth_y = self.current_state.get("y", 0.0)

        print(f"[NoitaEnv] Episode {self.episode_num} started.")
        return self._get_obs(), {}

    # ── Step ──────────────────────────────────────────────────────────────

    def step(self, action):
        # Send chosen action to Noita
        if self.websocket and self.loop:
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(json.dumps(int(action))),
                self.loop,
            )

        time.sleep(0.05)   # wait ~3 Noita frames for the action to take effect

        if self.current_state is None:
            return self._get_obs(), 0.0, False, False, {}

        s          = self.current_state
        current_y  = s.get("y",  0.0)
        current_hp = s.get("hp", 0.0)
        dead       = s.get("dead", False)

        # ── Reward ────────────────────────────────────────────────────────
        reward = 0.0

        # 1. Survival: small constant reward for staying alive
        reward += 0.01

        # 2. Depth progress: one-time bonus for reaching a new depth record
        #    (Noita Y axis: positive = deeper underground)
        if current_y > self.max_depth_y:
            reward += (current_y - self.max_depth_y) * 0.3
            self.max_depth_y = current_y

        # 3. Damage penalty: proportional to HP lost this step
        if current_hp < self.last_hp:
            reward -= (self.last_hp - current_hp) * 10.0

        # 4. Death penalty
        if dead:
            reward -= 2.0

        self.last_hp         = current_hp
        self.episode_steps  += 1
        self.episode_reward += reward

        if dead:
            print(
                f"[NoitaEnv] Ep {self.episode_num:3d} done — "
                f"steps={self.episode_steps:5d}  "
                f"reward={self.episode_reward:8.2f}  "
                f"max_depth={self.max_depth_y:.0f}"
            )
            # Clear state so reset() waits for a fresh one from the respawned player
            self.current_state = None

        return self._get_obs(), reward, dead, False, {}

    def render(self):
        pass

    def close(self):
        pass
