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

## Action space (MultiDiscrete `[3, 2, 2, 2, 10]`)

Five independent heads. PPO factorises log-probabilities head-by-head, so the
policy can express *combinations* like "move right + jump + fire up + kick" in
a single step. This replaced the prior `Discrete(18)` space, which forced a
single choice per frame and trained the agent into a kick-only local optimum
because moving and shooting were mutually exclusive.

| Index | Head    | Values | Effect |
|-------|---------|--------|--------|
| 0     | move    | 0=Idle, 1=Left, 2=Right        | vx target (−60 / 0 / +60) |
| 1     | jump    | 0=Off, 1=On                    | vy=−150 when grounded; no-op airborne |
| 2     | jetpack | 0=Off, 1=On                    | Δvy=−12/tick, burns mFlyingTimeLeft |
| 3     | kick    | 0=Off, 1=On                    | 0.5 melee damage in 30 px hemisphere, 15-frame cooldown |
| 4     | wand    | 0=Idle, 1=AutoAim, 2..9=8 dirs | 0: no fire; 1: aim+fire at nearest enemy (250 px); 2..9: R, UR, U, UL, L, DL, D, DR |

No smoothing: `mVelocity.x` is set to the target every action tick. This
tightens credit assignment but means the agent can stop on a dime.

**Aim writing is unconditional whenever wand ≥ 1.** A previous "camera-jitter
fix" debounced aim writes; it accidentally froze the cursor entirely and is
gone. The trade-off (more aggressive camera motion when the agent rapidly
switches wand direction) is accepted to make the agent actually able to aim.

## Reward function

```
−0.001                              time tax (small)
+0.015 × Δmanhattan_from_spawn       new max distance from spawn
+0.5                                 per newly visited 32×32 chunk (skipped if sky_visibility ≥ 0.3 — anti sky-farm)
+0.05  × Δdepth_y                    new depth record
+0.005 × Δy   (when moving down)     dense downward bias every step that moves deeper
+0.5                                 wand-fire visibly extended a ray ≥45 px (terrain destroyed)
+1.0                                 perfect_aim hitscan + rising-edge of fire + wand_ready (one shot launched at an enemy along an unobstructed line)
−0.02                                wand fired with no enemy visible OR wand on cooldown (waste)
−1.0 × Δhp                           damage taken this step
+5.0 × Δkills    (wand-attributed)   enemies killed via wand
+1.0 × Δkills    (kick-attributed)   enemies killed within 15 game frames of a kick — nerfed
+3.0                                 per chest opened
+(0.3 − portal_dist) × 0.05          pull toward Holy Mountain portal when within 30% of range
+20.0                                one-shot detected portal teleport (Δpos > 300 px AND was near portal last step)
+50 × biome_idx                      first time entering each new biome (coalmine=50, snowcave=100, …, the_work=300)
+100                                 per orb collected (proxy for boss defeat)
−1.0                                 on death
truncate −2 after 500 steps without descent
truncate −3 after 80% pure-idle in 500 steps (action loop)
```

**Kill attribution:** `init.lua` echoes `last_kick_frame` in the state JSON.
Python compares `(state.frame − state.last_kick_frame) <= 15` to decide
whether a kill in this step is wand- or kick-attributed.

**Perfect aim:** `init.lua` raycasts the wand's aim direction with
`RaytraceSurfacesAndLiquiform` for wall distance, then projects each enemy
onto the ray; `perfect_aim=true` if any enemy sits before the wall within
14 px perpendicular tolerance. The +1.0 reward is gated on rising-edge fire
+ `wand_ready` so the agent can't farm by holding the trigger.

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
- **Jetpack and jump are separate heads of the MultiDiscrete action.** The
  jump head only takes effect while `on_ground` (airborne jump is a no-op).
  The jetpack head adds Δvy = −12/tick and burns `mFlyingTimeLeft` by 8/tick;
  it works in mid-air until fuel runs out. The classic sky-farm exploit
  (fly up, collect chunk reward, repeat) is blocked on the Python side by
  gating `+0.5 chunk reward` on `sky_visibility < 0.3` — visiting open sky
  pays nothing.
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
- **Video Recording Quality**: Need to improve the quality of video recordings (TODO).

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

## Dev-notes protocol (mandatory)

Every development session, architectural change, bug fix, or experiment MUST be logged.

**Rules (enforced):**
- Create a file in `/dev_notes/YYYY-MM-DD.md` before making any code change.
- Document: what you intend to change, why, and the expected outcome.
- After making changes, append: what actually changed and initial results/observations.
- You are **forbidden** from committing or applying code changes without a corresponding dev_notes entry for that session.

File naming: `dev_notes/2026-05-17.md`, `dev_notes/2026-05-18.md`, etc.

---

## Azure Telemetry Pipeline

The training pipeline ships a zero-blocking, asynchronous telemetry system (`azure_telemetry.py`).

### What is logged
- **Per step (buffered in RAM):** session_id, episode, global_step, obs (60 floats), action, reward, done
- **Per episode (flushed at end):** full step buffer as compressed JSONL → Azure Blob Storage; episode summary JSON → Azure Cosmos DB
- **Assets:** model checkpoints (.zip), GIF recordings → Azure Blob Storage (`noita-assets` container)

### Azure services used (free-tier safe)
| Service | Usage | Free tier |
|---------|-------|-----------|
| Azure Cosmos DB (serverless) | Episode summary documents | 1000 RU/s, 25 GB |
| Azure Blob Storage | Step JSONL.gz + checkpoints + GIFs | 5 GB LRS |

### Setup
Add to your `.env` file:
```env
AZURE_COSMOS_URL=https://your-account.documents.azure.com:443/
AZURE_COSMOS_KEY=your-primary-key==
AZURE_COSMOS_DB=noitarl
AZURE_COSMOS_CONTAINER=episodes
AZURE_BLOB_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_BLOB_CONTAINER_STEPS=noita-steps
AZURE_BLOB_CONTAINER_ASSETS=noita-assets
```

Install SDK:
```bash
pip install azure-cosmos azure-storage-blob
```

When credentials are absent, telemetry silently no-ops — training proceeds normally.

### Architecture
```
NoitaMonitorCallback._on_step()
  ├─ telemetry.log_step(step_data)     # O(1), queue.put, never blocks
  └─ (episode end) telemetry.flush_episode(info)  # queues work, never blocks

AzureTelemetry._worker (background thread)
  ├─ batch JSONL.gz → BlobServiceClient.upload_blob()
  ├─ episode doc    → CosmosClient.upsert_item()
  └─ asset upload   → BlobServiceClient.upload_blob()
```

---

## Dependencies

```
Python 3.11+
gymnasium
stable-baselines3[extra]
websockets
numpy
azure-cosmos
azure-storage-blob
```

Install:
```bash
pip install stable-baselines3[extra] websockets gymnasium azure-cosmos azure-storage-blob
```
