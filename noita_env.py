import gymnasium as gym
import numpy as np
import asyncio
import json
import websockets
import threading
import time

class NoitaEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, host="localhost", port=5001):
        super(NoitaEnv, self).__init__()
        
        # Action space: 0:none, 1:left, 2:right, 3:jump, 4:fire
        self.action_space = gym.spaces.Discrete(5)
        
        # Observation space: 16 rays, HP, VX, VY, Mana(placeholder)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(20,), dtype=np.float32)
        
        self.host = host
        self.port = port
        self.current_state = None
        self.last_y = None
        self.last_hp = 1.0
        
        self.loop = None
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        
        print(f"Waiting for Noita to connect on {host}:{port}...")

    def _run_server(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        async def main():
            async with websockets.serve(self._handle_connection, self.host, self.port):
                await asyncio.Future()  # run forever

        self.loop.run_until_complete(main())

    async def _handle_connection(self, websocket):
        print("Noita connected!")
        self.websocket = websocket
        try:
            async for message in websocket:
                self.current_state = json.loads(message)
        except websockets.exceptions.ConnectionClosed:
            print("Noita disconnected")
            self.websocket = None

    def _get_obs(self):
        if self.current_state is None:
            return np.zeros((20,), dtype=np.float32)
        
        s = self.current_state
        obs = np.array(s['rays'] + [
            s['hp'],
            np.clip(s['vx'] / 200, -1, 1) * 0.5 + 0.5, # Normalize velocity to 0-1
            np.clip(s['vy'] / 200, -1, 1) * 0.5 + 0.5,
            0.0 # Mana placeholder
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        print("Reset called, waiting for connection/data...")
        # Wait for Noita to connect and send first state
        while self.current_state is None:
            time.sleep(0.5)
            if not self.server_thread.is_alive():
                print("Warning: Server thread died!")
            
        print("Data received! Starting episode.")
        self.last_y = self.current_state['y']
        self.last_hp = self.current_state['hp']
        
        return self._get_obs(), {}

    def step(self, action):
        if hasattr(self, 'websocket') and self.websocket:
            # print(f"Sending action: {action}")
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(json.dumps(int(action))), 
                self.loop
            )
        else:
            print("Warning: Step called but no websocket connection!")

        # Wait for next state
        time.sleep(0.05)
        
        if self.current_state is None:
            return self._get_obs(), 0, False, False, {}
            
        obs = self._get_obs()
        
        # Reward calculation
        current_y = self.current_state['y']
        current_hp = self.current_state['hp']
        
        # + Progress downward
        reward = (current_y - self.last_y) * 0.1
        
        # + Survival
        reward += 0.01
        
        # - Damage penalty
        if current_hp < self.last_hp:
            reward -= 50.0
            
        self.last_y = current_y
        self.last_hp = current_hp
        
        terminated = current_hp <= 0
        truncated = False # Add timeout if needed
        
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass
