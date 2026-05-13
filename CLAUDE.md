# noitarl — Reinforcement Learning agent for Noita

## What this is

A real-time RL bridge between **Noita** (the game) and a **PPO neural network** (Stable Baselines3).
The mod runs inside Noita as a Lua script, gathers game state every frame via raycasts, sends it
over a local WebSocket, receives a discrete action back, and applies it by writing directly to the
player entity's velocity component.

Goal: train a bot that learns to survive and descend deeper into the procedurally generated world.

---

## Repository layout

```
mods/noitarl/
├── init.lua            Noita mod entry point (Lua, runs inside the game)
├── noita_env.py        Gymnasium environment (Python, WebSocket server)
├── train.py            Single-env PPO training script
├── train_multi.py      Multi-env parallel training (SubprocVecEnv)
├── port.txt            WebSocket port this Noita instance listens on (default 5001)
├── lib/
│   ├── pollnet.lua     LuaJIT FFI WebSocket bindings
│   └── json.lua        Lua JSON encoder/decoder
├── bin/
│   └── pollnet.dll     Native WebSocket DLL (Windows x64)
├── docs/
│   ├── lua_api_documentation.txt    Noita Lua API reference
│   └── component_documentation.txt  All entity component fields
├── CLAUDE.md           ← you are here
└── ROADMAP.md          Development plan
```

---

## Architecture

```
Noita (game, 60 fps)                    Python process
─────────────────────                   ───────────────
OnWorldPreUpdate()                      NoitaEnv.step(action)
  └─ apply_action(pending_action)         ├─ send action JSON  ──────────────┐
        └─ CharacterDataComponent          │                                  │
             .mVelocity = (±60, vy)        │                                  ↓
                                           │                         WebSocket (localhost)
OnWorldPostUpdate()                        │                                  │
  ├─ build state (rays, hp, vx, vy…)       │                                  │
  ├─ detect death → respawn               │                                  │
  ├─ send state JSON ────────────────────→│                                  │
  └─ recv action ←────────────────────────┘
        └─ pending_action = action
```

### Why direct velocity, not ControlsComponent buttons

`ControlsComponent.mButtonDownLeft/Right/Jump` are **overwritten by the engine's keyboard reader
every frame** before the physics tick. Setting them in `OnWorldPreUpdate` has no effect.
Writing to `CharacterDataComponent.mVelocity` bypasses that pipeline and is consumed directly
by `PlayerCollisionSystem`.

---

## Observation space (60 float32, all in [0, 1])

| Index   | Feature | Notes |
|---------|---------|-------|
| 0–15    | 16 platform rays   | RaytracePlatforms; 0 = wall, 1 = 150 px clear |
| 16–23   | 8 enemy radar      | 1 = none, 0 = enemy at player (200 px range) |
| 24–31   | 8 liquid sensors   | 0 = dry, ~1 = liquid pool ahead (80 px) |
| 32–39   | 8 projectile radar | 1 = clear, 0 = bullet at player (150 px) |
| 40–47   | 8 gold radar       | 1 = no gold, 0 = gold at player (150 px) |
| 48      | hp fraction        | virtual HP (4 engine units), 0 = dead |
| 49      | vx normalised      | (−200..+200) → (0..1) |
| 50      | vy normalised      | positive vy = falling |
| 51      | on_ground          | 0 or 1 |
| 52      | jetpack fuel       | 0 = empty, 1 = full |
| 53      | wand ready         | 0 = on cooldown, 1 = can fire |
| 54      | is_on_fire         | 0 or 1; from DamageModelComponent.is_on_fire |
| 55      | is_poisoned        | 0 or 1; any of RADIOACTIVE/POISONED/STAINED_RADIOACTIVE |
| 56      | sky_visibility     | 1 = surface, 0 = deep underground (depth proxy) |
| 57      | portal distance    | 1 = no portal in 400 px, 0 = at portal |
| 58      | portal dx_norm     | 0.5 = same X; <0.5 portal is left, >0.5 portal is right |
| 59      | portal dy_norm     | 0.5 = same Y; <0.5 portal is above, >0.5 portal is below |

## Action space (Discrete 10)

Composable move+jump so the agent can hop over corridor ledges. Fire is a
separate action; action 7 forces aim straight down ("dig"). KICK is a
short-range melee that bypasses wand cooldown. JETPACK_HOLD burns fuel
for sustained ascent.

| Value | Effect |
|-------|--------|
| 0 | IDLE          — vx target = 0, no jump, no fire |
| 1 | LEFT          — vx = −60 |
| 2 | RIGHT         — vx = +60 |
| 3 | JUMP          — vy = −150 (only when on_ground; no-op airborne) |
| 4 | LEFT+JUMP     — vx = −60 AND jump (vertical part only triggers grounded) |
| 5 | RIGHT+JUMP    — vx = +60 AND jump (vertical part only triggers grounded) |
| 6 | FIRE          — auto-aim at nearest enemy, fire wand |
| 7 | FIRE_DOWN     — aim straight down, fire wand (used for digging) |
| 8 | KICK          — 0.5 melee damage in 30 px hemisphere ahead, 15-frame cooldown |
| 9 | JETPACK_HOLD  — Δvy = −12/tick, burns mFlyingTimeLeft (no horizontal effect) |

No smoothing: `mVelocity.x` is set to the target every action tick. This
tightens credit assignment but means the agent can stop on a dime.

## Reward function

```
−0.001                            time tax (small)
+0.02 × Δmanhattan_from_spawn      when a new max distance from spawn is reached
+0.5                               per newly visited 32×32 chunk (skipped if sky_visibility ≥ 0.3 — anti sky-farm)
+0.02 × Δdepth_y                   small bonus for new depth record
−1.0  × Δhp                        damage taken this step
+5.0  × Δkills                     enemies killed
+0.05                              FIRE/FIRE_DOWN pressed AND enemy in 250 px (aim-on-enemy bonus)
+(0.3 − portal_dist) × 0.05        gradient pulling agent into portal (only when within 30% of PORTAL_RANGE)
+20.0                              one-shot bonus on detected portal teleport (Δpos > 300 px AND was near portal last step)
−1.0                               on death
truncate after 600 steps without progress  (no penalty — just ends episode)
```

The reward is built around Manhattan distance from the per-episode spawn,
because the old "+Δdepth only" reward punished horizontal corridors. The
−10 cowardice penalty has been removed entirely.

---

## How to run (single env)

```bash
cd "C:\Program Files (x86)\Steam\steamapps\common\Noita\mods\noitarl"
.venv\Scripts\activate          # or: venv\Scripts\activate
python train.py                 # starts WebSocket server, waits for Noita
# ── in another window ──
# Launch Noita with mod "noitarl" enabled
```

Live training curves:
```bash
tensorboard --logdir ./noita_ppo_tensorboard/
# open http://localhost:6006
```

## How to run (multi env)

Each Noita instance needs a unique port. Edit `port.txt` in each copy's mod folder:
- Instance A: `port.txt` = `5001`
- Instance B: `port.txt` = `5002`
- …

```bash
python train_multi.py --envs 2 --port 5001
```

> ⚠️ **Save directory isolation is mandatory.** Two Noita instances sharing the same
> `%APPDATA%\LocalLow\Nolla_Games_Noita\` folder will corrupt each other's world files
> (trees flying, missing floor tiles, physics explosions). Each instance must run from
> a **separate copy of the Noita installation directory** so saves go to different paths.
> If isolation is impossible, use single-env `train.py` instead.

> ⚠️ **Overfitting to one world seed.** The mod uses teleport-respawn so Noita never
> restarts during a training session. The agent learns the same procedurally-generated
> world every episode. This is expected: the policy learns to exploit a single layout,
> which looks impressive for demos but won't generalise to a new seed. If generalization
> matters, restart Noita periodically (which triggers a world re-roll) and let the
> `connection_retry_interval` reconnect logic handle the gap.

Resume from checkpoint:
```bash
python train_multi.py --envs 2 --resume checkpoints/noita_ppo_2env_400000_steps.zip
```

---

## Tips for future agents working on this codebase

- **`ent_coef = 0.02` is mandatory** in `config.py`. SB3's default `0.0`
  produces deterministic policies that stop exploring after ~10k steps.
  Symptom: agent stands still or twitches in spawn area for hours.
- **The "virtual HP" hack** (`init.lua: IMMORTAL_HP = 10000`, `VIRTUAL_MAX_HP = 4`)
  keeps Noita's engine from killing the player entity. When `virtual_hp` hits 0
  we teleport-respawn instead of triggering a real death sequence. This exists
  because Noita has no level-reset API — we can't reload the seed without
  restarting the game. *Why not just disable DamageModelComponent?* Because the 
  engine natively handles damage numbers, stains, and physics reactions through it. 
  We need it active to accurately read damage taken while intercepting the fatal blow.
- **Spawn randomization** uses a `spawn_candidates` pool with `±30 px` X jitter
  to prevent the policy from overfitting to one corridor entrance.
- **Initial descent into the Mines.** The surface biome has no corridor
  structure, so the agent just wanders. On the first frame after connect we
  raycast straight down from the surface spawn and teleport the player onto
  the first platform below (`INITIAL_DESCENT_RANGE = 1500 px`,
  `INITIAL_DESCENT_LIFT = 20 px`). That post-descent position is what
  `spawn_candidates` records — every subsequent respawn lands underground.
- **Jetpack is action 9 (JETPACK_HOLD).** Bare JUMP (3/4/5) still only fires
  when grounded; airborne JUMP is a no-op. Action 9 adds Δvy = −12/tick and
  burns `mFlyingTimeLeft` by 8/tick. The classic sky-farm exploit (fly up,
  collect chunk reward, repeat) is blocked on the Python side by gating
  `+0.5 chunk reward` on `sky_visibility < 0.3` — visiting open sky pays
  nothing.
- **FRAME_SKIP = 4**. We process an action and send state every 4 frames (15 times 
  a second at 60 FPS). This is a trade-off: it reduces CPU overhead and network 
  latency while being fast enough for the agent to react to Noita's physics. 
  Without frame skip, the agent struggles to correlate actions with delayed physics 
  consequences.
- **Action trace log**: every applied action is appended as one JSON line to
  `actions_trace.jsonl` (5 MB rotation). Useful for verifying that the agent's
  intended action actually became a `vx/vy` change on the physics tick.
- **Each game restart re-rolls the world seed.** Long-term curricula must
  account for this — there is no "save and resume in same level" path.

## Key Noita API facts (for future agents)

- **Callback hooks**: `OnWorldPreUpdate()` (before physics), `OnWorldPostUpdate()` (after physics).
  Both are valid global functions in `init.lua`.
- **Entity tag for player**: `"player_unit"` → `EntityGetWithTag("player_unit")[1]`
- **Reading component fields**: `ComponentGetValue2(component_id, "field_name")` — works for both
  Members and Privates. Returns nil if field doesn't exist (silent failure, no error).
- **Writing component fields**: `ComponentSetValue2(component_id, "field_name", value)`
- **Raytrace function**: `RaytracePlatforms(x1,y1,x2,y2) → hit, hx, hy` — stops on solid platforms.
  Note: it is **RaytracePlatforms**, NOT RaycastPlatforms (that function does not exist).
- **Coordinate system**: +X = right, +Y = **down** (larger Y = deeper underground).
- **Lua runtime**: LuaJIT (FFI available, `require` is restricted — use `dofile` for local libs).
- **`ComponentSetValue2` on `CharacterDataComponent.mVelocity`**: takes two floats `(vx, vy)`.
- **`DEBUG_MARK`** is available in dev builds only. It renders heavy 3D markers — avoid using it
  in per-frame loops (kills FPS). Use GuiText + GuiColorSetForNextWidget for HUD overlays instead.

---

## Dependencies

```
Python 3.11+
gymnasium
stable-baselines3[extra]
websockets
numpy
```

Install:
```bash
pip install stable-baselines3[extra] websockets gymnasium
```
