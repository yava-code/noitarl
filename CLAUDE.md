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

## Observation space (20 float32, all in [0, 1])

| Index | Feature | Notes |
|-------|---------|-------|
| 0–15  | 16 raytrace sensors | angle = i × π/8; 0 = wall at player, 1 = 150 px clear |
| 16    | hp fraction | 0 = dead, 1 = full health |
| 17    | vx normalised | (−200..+200) → (0..1) |
| 18    | vy normalised | positive vy = falling |
| 19    | on_ground | 0 or 1 |

## Action space (Discrete 5)

| Value | Effect |
|-------|--------|
| 0 | IDLE — no input |
| 1 | LEFT — vx = −60 |
| 2 | RIGHT — vx = +60 |
| 3 | JUMP — vy = −150 (only when on_ground) |
| 4 | FIRE — mButtonDownFire = true |

## Reward function

```
+0.01          per step (survival)
+0.3 × Δdepth  when player reaches a new Y record (Noita Y increases downward)
−10 × Δhp      proportional to damage taken this step
−2.0           on death
```

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

Resume from checkpoint:
```bash
python train_multi.py --envs 2 --resume checkpoints/noita_ppo_2env_400000_steps.zip
```

---

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
