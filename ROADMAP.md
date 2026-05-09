# noitarl — Development Roadmap

## Current state (v0.1 — MVP)

- [x] WebSocket bridge Noita ↔ Python (pollnet.dll + LuaJIT FFI)
- [x] 16-ray observation + HP + velocity + on_ground
- [x] 5 discrete actions via direct velocity injection
- [x] PPO training (Stable Baselines3)
- [x] Episode reset: death detection + in-place respawn
- [x] Lightweight HUD radar (no FPS impact)
- [x] Multi-env training script (train_multi.py)
- [x] Checkpoint saving every 100k steps

---

## Phase 1 — Better observations & reward  *(~1 week)*

**Goal:** give the agent enough signal to learn non-trivial behaviour.

- [ ] **Velocity history** — include last 3 frames of vx/vy (6 extra features) so the agent
      can infer acceleration and predict trajectory.
- [ ] **Enemy proximity sensors** — separate raytrace pass checking for enemy entities in
      8 directions using `EntityGetInRadiusWithTag("enemy", ...)`.
- [ ] **Reward shaping**
  - Add kill reward: compare enemy count before/after each step using `EntityGetWithTag`.
  - Add wand/item pickup reward: detect `EntityGetWithTag("item_pickup")` events.
  - Reduce depth reward weight once agent reliably descends (prevents pure rushing).
- [ ] **HP normalisation fix** — current formula assumes max_hp is constant; add safety clamp
      for polymorph effects that change max_hp mid-episode.

---

## Phase 2 — Stable episodes  *(~1 week)*

**Goal:** episodes that end cleanly and restart consistently.

- [ ] **Auto-respawn at world warp point** — use `EntityLoad("data/entities/player.xml")`
      instead of restoring HP in-place; prevents getting stuck in walls.
- [ ] **Episode timeout** — if no depth progress for N steps, force reset (prevents infinite
      loops of the agent standing still).
- [ ] **Python-side episode stats** — log mean reward, mean depth, episode length per episode
      to a CSV file for post-training analysis.
- [ ] **TensorBoard custom scalars** — push episode_reward and max_depth as custom metrics
      using `model.logger.record()` in a custom callback.

---

## Phase 3 — Architecture improvements  *(~2 weeks)*

**Goal:** better network, faster convergence.

- [ ] **Frame stacking** — wrap env with `VecFrameStack(env, n_stack=4)` so the policy sees
      the last 4 observations. Helps with velocity/momentum inference.
- [ ] **Larger network** — replace default `MlpPolicy` with `policy_kwargs=dict(net_arch=[256,256])`
      for more expressive representations.
- [ ] **Curriculum learning** — start with a shallow world (mod the biome config to flatten the
      first level) and gradually increase difficulty.
- [ ] **Prioritised experience** — switch from PPO to SAC or TD3 with a replay buffer for
      better sample efficiency (needs continuous action space adaptation).

---

## Phase 4 — Multi-agent & server  *(~2 weeks)*

**Goal:** 4–8× faster data collection.

- [ ] **Headless Noita on Linux** — run Noita under Wine + Xvfb on a GPU server.
      Requires testing Wine compatibility with the dev build.
- [ ] **Docker image** — package Noita + mod + Python env into a container for reproducibility.
- [ ] **Dynamic port assignment** — Lua reads a lock file to auto-select a free port so N
      instances can share one mods folder without manual port.txt edits.
- [ ] **Async state collection** — replace `time.sleep(0.05)` in `step()` with an asyncio event
      so the Python side truly waits for the next Noita frame, eliminating stale-state reads.

---

## Phase 5 — YouTube-ready agent  *(~1 month of training)*

**Goal:** visually compelling, demonstrably intelligent behaviour.

- [ ] **Polish HUD** — add a live reward graph drawn in the HUD using multiple GuiText rows
      (simple bar chart of last 30 episode rewards).
- [ ] **Evaluation script** `eval.py` — load a saved model and run it in deterministic mode
      (`model.predict(obs, deterministic=True)`) for clean recording sessions.
- [ ] **Slowdown mode** — add a Noita mod setting to run at 30 % speed for dramatic effect
      (record gameplay at reduced tick rate via `PHYSICS_FPS_CAP` in magic_numbers.xml).
- [ ] **Overlay stats** — show neural network activation heatmap (which rays are "hot") by
      extracting first-layer weights and displaying them on the radar grid.
- [ ] **Milestone demo** — record separate clips at 50k / 200k / 500k / 1M steps to show
      the progression from random twitching to strategic descent.

---

## Known issues / tech debt

| Issue | Impact | Fix |
|-------|--------|-----|
| `time.sleep(0.05)` in step() | ~3-frame state lag | Replace with asyncio event wait |
| `on_ground` check in Lua before respawn | Can miss death if entity is removed first | Add entity existence guard |
| No wand/spell state in observation | Agent can't learn to use magic | Add wand type as categorical feature |
| Mana placeholder (was always 0) | Wasted feature slot | Now `on_ground` — still not mana |
| Fire action doesn't aim | Shoots in random direction | Set `mMousePosition` in ControlsComponent |

---

## Experiment ideas

- **Negative reward for shooting** (encourage efficient resource use) — toggle via `reward_fire_penalty` flag.
- **Multi-head policy** — separate network branch for aiming vs movement.
- **Intrinsic curiosity module (ICM)** — bonus reward for visiting novel states (helps with sparse reward early on).
- **Imitation learning pre-training** — record human play sessions → behavioural cloning → fine-tune with PPO.
