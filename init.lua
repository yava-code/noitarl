-- mods/noitarl/init.lua
-- Reinforcement Learning bridge — Noita side.
-- Communicates with noita_env.py via WebSocket (pollnet.dll).

-- ── Logging ───────────────────────────────────────────────────────────────
local function script_dir()
    local source = debug.getinfo(1, "S").source or ""
    if source:sub(1, 1) == "@" then
        source = source:sub(2)
    end
    source = source:gsub("\\", "/")
    return source:match("^(.*)/[^/]+$") or "."
end

local MOD_ROOT = script_dir()
local function mod_path(rel)
    return MOD_ROOT .. "/" .. rel
end

local LOG_FILE = mod_path("logger.txt")

local function log(level, msg)
    local line = string.format("[%s] [%s] %s", os.date("%H:%M:%S"), level, tostring(msg))
    print(line)
    local f = io.open(LOG_FILE, "a")
    if f then f:write(line .. "\n"); f:close() end
end
local function info(m)  log("INFO ", m) end
local function warn(m)  log("WARN ", m) end
local function err(m)   log("ERROR", m) end

-- Clear log on each start so it doesn't grow forever
do local f = io.open(LOG_FILE, "w"); if f then f:close() end end
info("Mod init started — noitarl v0.4 (no jetpack, auto-descent into Mines)")

-- ── Safe component accessors ──────────────────────────────────────────────
-- Noita's ComponentGetValue2 / ComponentSetValue2 silently crash if the
-- component or field doesn't exist, so we wrap them.
local function cget(comp, field)
    if not comp then return nil end
    local ok, v1, v2 = pcall(ComponentGetValue2, comp, field)
    if not ok then warn("cget " .. field .. ": " .. tostring(v1)); return nil end
    return v1, v2
end

local function cset(comp, field, ...)
    if not comp then return false end
    local ok, e = pcall(ComponentSetValue2, comp, field, ...)
    if not ok then warn("cset " .. field .. ": " .. tostring(e)); return false end
    return true
end

-- ── Libraries ─────────────────────────────────────────────────────────────
local json_ok, json = pcall(dofile, mod_path("lib/json.lua"))
if not json_ok then err("json.lua failed: " .. tostring(json)); return end

local pn_ok, pollnet = pcall(dofile, mod_path("lib/pollnet.lua"))
if not pn_ok then err("pollnet.lua failed: " .. tostring(pollnet)); return end

-- ── Config ────────────────────────────────────────────────────────────────
local function read_port()
    local f = io.open(mod_path("port.txt"), "r")
    if f then local p = tonumber(f:read("*l")); f:close(); if p then return p end end
    return 5001
end
local WS_PORT = read_port()
local WS_URL  = "ws://localhost:" .. WS_PORT
info("Port: " .. WS_PORT)

-- ── Connection state ──────────────────────────────────────────────────────
local socket                    = nil
local last_connection_attempt   = 0
local connection_retry_interval = 180    -- ~3 s at 60 fps
local consecutive_errors        = 0
local MAX_ERRORS                = 5      -- give up and reconnect after N errors
local gui                       = nil

-- ── Movement constants ────────────────────────────────────────────────────
local MOVE_SPEED     = 60
local JUMP_SPEED     = -150
local JETPACK_DV     = 12     -- vy delta per tick while JETPACK_HOLD action is on
local JETPACK_MAX_VY = -200   -- terminal upward velocity (clamp)
local JETPACK_FUEL_BURN = 8   -- mFlyingTimeLeft units burned per tick
local KICK_RANGE     = 30     -- px from player to count enemies for KICK
local KICK_DAMAGE    = 0.5    -- engine HP units (≈ 12% of player max)
local KICK_COOLDOWN  = 15     -- frames between KICK actions to prevent DPS-spam
local CHEST_RANGE    = 22     -- px: if any chest centre is within this radius → auto-open
local FRAME_SKIP     = 4      -- Process action/state every 4 frames (trade-off: reduces CPU overhead while keeping fast 15 FPS reactions)

-- Bare JUMP (action 3/4/5) still requires on_ground; only the explicit JETPACK_HOLD
-- (action 9) burns fuel for sustained ascent. The sky-farm exploit (flying to the
-- ceiling to harvest chunk bonus) is countered on the Python side by gating chunk
-- reward on sky_visibility < 0.3.

-- ── Initial descent: drop the player from the surface into the Mines on ──
-- the very first frame, then make THAT the spawn point. The surface biome
-- is open and lacks the narrow corridors that make Noita interesting; if
-- the agent is left on the surface it wanders aimlessly. We raycast
-- straight down and land just above the first platform we hit.
local INITIAL_DESCENT_RANGE  = 1500   -- max px below surface to search
local INITIAL_DESCENT_LIFT   = 20     -- gap above the platform we land on
local initial_descent_done   = false

-- Immortal-agent HP hack: engine never kills the player;
-- we track "virtual HP" ourselves and teleport-respawn when it hits 0.
-- Why not disable DamageModelComponent? Because the engine handles damage numbers, 
-- stains, and physics reactions through it natively.
local IMMORTAL_HP    = 10000.0
local VIRTUAL_MAX_HP = 4.0    -- 4.0 engine units ≈ 100% HP in UI
local virtual_hp     = VIRTUAL_MAX_HP

-- ── Chest session counters ────────────────────────────────────────────────
local chests_opened_total = 0    -- across all episodes this session
local chests_opened_ep    = 0    -- reset each episode

-- ── Per-frame state ───────────────────────────────────────────────────────
-- Discrete 10 action space:
--   0 IDLE | 1 LEFT | 2 RIGHT | 3 JUMP | 4 L+JUMP | 5 R+JUMP
--   6 FIRE | 7 FIRE_DOWN | 8 KICK | 9 JETPACK_HOLD
local pending_action  = 0
local last_action     = 0
local last_facing     = 1     -- last non-zero horizontal direction (for KICK aim)
local last_kick_frame = -1000
local ACTION_NAMES    = {
    [0]="IDLE", [1]="LEFT", [2]="RIGHT", [3]="JUMP",
    [4]="L+JMP", [5]="R+JMP", [6]="FIRE", [7]="DIG_D",
    [8]="KICK", [9]="JETPK",
}
-- Decomposition: {move_x, do_jump, do_fire, aim_down, do_kick, do_jetpack}
local ACTION_DECODE = {
    [0]={ 0, false, false, false, false, false },
    [1]={-1, false, false, false, false, false },
    [2]={ 1, false, false, false, false, false },
    [3]={ 0, true,  false, false, false, false },
    [4]={-1, true,  false, false, false, false },
    [5]={ 1, true,  false, false, false, false },
    [6]={ 0, false, true,  false, false, false },
    [7]={ 0, false, true,  true,  false, false },
    [8]={ 0, false, false, false, true,  false },
    [9]={ 0, false, false, false, false, true  },
}
local MAX_EP_STEPS    = 4000  -- ~4.5 min at 60 fps with FRAME_SKIP=4

-- ── Episode tracking ──────────────────────────────────────────────────────
-- spawn_candidates accumulates good "anchor" positions; respawn picks one at random
local spawn_x, spawn_y      = nil, nil   -- recorded on first frame
local spawn_candidates      = {}         -- list of {x=, y=}
local SPAWN_JITTER          = 30         -- Random ± X jitter to prevent policy from overfitting to a single corridor start
local episode_num           = 0
local episode_steps         = 0
local frame_times           = {}   -- rolling window for FPS estimate
local PERF_WINDOW           = 60

-- Action trace log (one line per applied action) for offline debugging
local LOG_FILE_ACTIONS      = mod_path("actions_trace.jsonl")
local ACTION_LOG_ROTATE_AT  = 5 * 1024 * 1024   -- 5 MB
local action_log_size       = 0
do local f = io.open(LOG_FILE_ACTIONS, "w"); if f then f:close() end end

-- ── Action trace logger (offline debugging) ──────────────────────────────
local function log_action_trace(rec)
    local ok, line = pcall(json.encode, rec)
    if not ok then return end
    local f = io.open(LOG_FILE_ACTIONS, "a")
    if not f then return end
    f:write(line, "\n")
    f:close()
    action_log_size = action_log_size + #line + 1
    if action_log_size > ACTION_LOG_ROTATE_AT then
        local g = io.open(LOG_FILE_ACTIONS, "w"); if g then g:close() end
        action_log_size = 0
    end
end

-- ── Apply action: direct velocity injection, composable move+jump+fire ──
-- No smoothing — grip is effectively 1.0 so the next physics tick sees the
-- intended target velocity. This tightens credit assignment for PPO.
local function apply_action(player, action)
    if not player or player == 0 then return end
    last_action = action

    local decode = ACTION_DECODE[action] or ACTION_DECODE[0]
    local move_x, do_jump, do_fire, aim_down, do_kick, do_jetpack =
        decode[1], decode[2], decode[3], decode[4], decode[5], decode[6]

    if move_x ~= 0 then last_facing = move_x end

    -- Physics: write velocity directly (mVelocity bypasses Noita's input layer)
    local vx_out, vy_out, on_ground_now = 0, 0, false
    local cdata = EntityGetFirstComponent(player, "CharacterDataComponent")
    if cdata then
        local cur_vx, cur_vy = cget(cdata, "mVelocity")
        local on_ground      = cget(cdata, "is_on_ground")
        cur_vx = cur_vx or 0; cur_vy = cur_vy or 0
        on_ground_now = on_ground == true

        -- Horizontal: set target directly (no lerp). When the agent says "go
        -- right", it goes right on the very next physics tick.
        local target_vx = move_x * MOVE_SPEED
        local new_vx    = target_vx

        -- Vertical:
        --   Bare JUMP (3/4/5) only fires when grounded.
        --   JETPACK_HOLD (9) adds upward Δv every tick while fuel remains.
        local new_vy = cur_vy
        if do_jump and on_ground_now then
            new_vy = JUMP_SPEED
        end
        if do_jetpack then
            local fuel = cget(cdata, "mFlyingTimeLeft") or 0
            if fuel > 0 then
                new_vy = math.max(cur_vy - JETPACK_DV, JETPACK_MAX_VY)
                cset(cdata, "mFlyingTimeLeft", math.max(0, fuel - JETPACK_FUEL_BURN))
            end
        end

        cset(cdata, "mVelocity", new_vx, new_vy)
        vx_out, vy_out = new_vx, new_vy
    end

    -- KICK: melee damage to enemies within KICK_RANGE in front of player.
    -- Hand-rolled because Noita's player has no MeleeAttackComponent by default.
    if do_kick then
        local frame = GameGetFrameNum()
        if frame - last_kick_frame >= KICK_COOLDOWN then
            last_kick_frame = frame
            local pok, px, py = pcall(EntityGetTransform, player)
            if pok then
                local eok, ents = pcall(EntityGetInRadiusWithTag, px, py, KICK_RANGE, "enemy")
                if eok and ents then
                    for _, eid in ipairs(ents) do
                        local tok, ex, ey = pcall(EntityGetTransform, eid)
                        if tok then
                            -- Front-facing hemisphere: enemy must be on the
                            -- facing side OR within 12 px vertically (kick up/down).
                            if (ex - px) * last_facing >= -8 then
                                pcall(EntityInflictDamage, eid, KICK_DAMAGE, "melee",
                                    "kicked", "BLOOD", last_facing * 250, -50, player,
                                    ex, ey, 200)
                            end
                        end
                    end
                end
            end
        end
    end

    -- Aiming + firing on ControlsComponent
    local ctrl = EntityGetFirstComponent(player, "ControlsComponent")
    if ctrl then
        local tok, px, py = pcall(EntityGetTransform, player)
        if tok then
            local nx, ny
            if aim_down then
                -- Override: dig straight down (used by action 7 FIRE_DOWN)
                nx, ny = px, py + 50
            else
                -- Auto-aim: nearest enemy in 250 px, else face current movement
                local aok, enemies = pcall(EntityGetInRadiusWithTag, px, py, 250, "enemy")
                local face_x       = (move_x ~= 0) and move_x or 1
                nx, ny = px + 50 * face_x, py
                if aok and enemies and #enemies > 0 then
                    local nearest_d2 = math.huge
                    for _, eid in ipairs(enemies) do
                        local eok, ex, ey = pcall(EntityGetTransform, eid)
                        if eok then
                            local d2 = (ex - px)^2 + ((ey + 4) - py)^2
                            if d2 < nearest_d2 then
                                nearest_d2, nx, ny = d2, ex, ey + 4
                            end
                        end
                    end
                end
            end
            local dx, dy = nx - px, ny - py
            local len = math.sqrt(dx*dx + dy*dy)
            if len > 0.001 then
                local nxv, nyv = dx/len, dy/len
                -- ControlsComponent gets pummelled by the engine's input layer each
                -- frame; writing aim here is necessary but not sufficient.
                cset(ctrl, "mAimingVectorNormalized", nxv, nyv)
                cset(ctrl, "mAimingVector",           nxv * 40, nyv * 40)
                cset(ctrl, "mMousePosition",          nx, ny)
                cset(ctrl, "mMousePositionRaw",       nx, ny)
                cset(ctrl, "mGamePadCursorInWorld",   nx, ny)
                cset(ctrl, "mGamepadIndirectAiming",  nxv, nyv)

                -- PlatformShooterPlayerComponent owns the smoothed aim vector that
                -- drives sprite flip + visible reticle. Without writing here, our
                -- ControlsComponent aim is overwritten by platformshooterplayer_system.
                local pspc = EntityGetFirstComponent(player, "PlatformShooterPlayerComponent")
                if pspc then
                    cset(pspc, "mSmoothedAimingVector",      nxv, nyv)
                    cset(pspc, "mDesiredCameraPos",          nx, ny)
                    cset(pspc, "mDesiredCameraPosForGfx",    nx, ny)
                end

                -- CharacterPlatformingComponent stores facing direction used by sprite.
                local cplat = EntityGetFirstComponent(player, "CharacterPlatformingComponent")
                if cplat then
                    cset(cplat, "mFacingDirection", (nxv >= 0) and 1 or -1)
                end
            end
        end

        -- Fire: rising-edge frame stamp so cast_delay isn't reset every frame
        local was_fire = (cget(ctrl, "mButtonDownFire") == true)
        cset(ctrl, "mButtonDownFire",      do_fire)
        cset(ctrl, "mButtonDownLeftClick", do_fire)
        if do_fire and not was_fire then
            local frame = GameGetFrameNum()
            cset(ctrl, "mButtonFrameFire",      frame)
            cset(ctrl, "mButtonFrameLeftClick", frame)
        end
    end

    -- Trace one record per applied action for offline analysis
    if cdata then
        log_action_trace({
            f  = GameGetFrameNum(),
            a  = action,
            mx = move_x, jp = do_jump and 1 or 0,
            fr = do_fire and 1 or 0, ad = aim_down and 1 or 0,
            vx = vx_out, vy = vy_out, gnd = on_ground_now and 1 or 0,
        })
    end
end

-- ── Respawn: teleport to (possibly randomised) spawn, reset state ────────
local function pick_spawn()
    -- Choose one of the recorded anchor positions and jitter it slightly so
    -- the agent doesn't overfit to a single corridor entrance.
    local n = #spawn_candidates
    if n == 0 then return spawn_x or 0.0, spawn_y or 0.0 end
    local sp = spawn_candidates[math.random(n)]
    local jx = math.random(-SPAWN_JITTER, SPAWN_JITTER)
    return sp.x + jx, sp.y
end

local function respawn_player(player)
    virtual_hp = VIRTUAL_MAX_HP

    local sx, sy = pick_spawn()
    local ok, e  = pcall(EntitySetTransform, player, sx, sy)
    if not ok then warn("EntitySetTransform failed: " .. tostring(e)) end

    local cdata = EntityGetFirstComponent(player, "CharacterDataComponent")
    if cdata then
        cset(cdata, "mVelocity", 0, 0)
        cset(cdata, "mFlyingTimeLeft", 1000.0)
    end

    -- Remove fire/toxic stain effects, then douse with water for good measure
    pcall(EntityRemoveStainStatusEffect,     player, "stain_fire")
    pcall(EntityRemoveStainStatusEffect,     player, "stain_radioactive_gas_1")
    pcall(EntityRemoveIngestionStatusEffect, player, "RADIOACTIVE")
    local water_id = CellFactory_GetType("water")
    pcall(EntityAddRandomStains, player, water_id, 400)

    episode_num       = episode_num + 1
    episode_steps     = 0
    pending_action    = 0
    chests_opened_ep  = 0
    info(string.format("Ep %d started — spawn (%.0f, %.0f) [pool=%d]",
        episode_num, sx, sy, #spawn_candidates))
end

-- ── Mini radar HUD (9 GuiText calls, zero world-space cost) ──────────────
-- Grid maps 8 directions; chars show wall proximity.
-- Ray angle = i*π/8.  Lua table is 1-indexed.
-- Direction → ray index:  E=1 SE=3 S=5 SW=7 W=9 NW=11 N=13 NE=15
local GRID = {
    { 11, 13, 15 },   -- NW  N  NE
    {  9, -1,  1 },   -- W   @   E
    {  7,  5,  3 },   -- SW  S  SE
}
local function dist_char(d)
    if d < 0.25 then return "#"
    elseif d < 0.5  then return "+"
    elseif d < 0.75 then return "."
    else return " " end
end

local function draw_radar(g, ox, oy, rays)
    local cell = 8
    for row = 1, 3 do
        for col = 1, 3 do
            local idx = GRID[row][col]
            local tx  = ox + (col - 1) * cell
            local ty  = oy + (row - 1) * cell
            if idx == -1 then
                GuiColorSetForNextWidget(g, 0, 1, 1, 1)
                GuiText(g, tx, ty, "@")
            else
                local d = rays[idx] or 1.0
                GuiColorSetForNextWidget(g, 1-d, d, 0, 1)
                GuiText(g, tx, ty, dist_char(d))
            end
        end
    end
    GuiColorSetForNextWidget(g, 1, 1, 1, 1)
end

-- ── 8-direction liquid sensors ────────────────────────────────────────────
-- Signal per ray: (d_solid - d_liquid) / LIQUID_LEN
--   0 = dry or no difference,  ~1 = liquid pool right in front
local LIQUID_LEN = 80

local function build_liquid_sensors(x, y)
    local out = {}
    for i = 0, 7 do
        local angle = i * (math.pi / 4)  -- 8 compass directions
        local tx = x + math.cos(angle) * LIQUID_LEN
        local ty = y + math.sin(angle) * LIQUID_LEN

        local ok1, hit1, hx1, hy1 = pcall(RaytraceSurfacesAndLiquiform, x, y, tx, ty)
        local ok2, hit2, hx2, hy2 = pcall(RaytracePlatforms,            x, y, tx, ty)

        local d1 = (ok1 and hit1) and math.sqrt((hx1-x)^2+(hy1-y)^2) or LIQUID_LEN
        local d2 = (ok2 and hit2) and math.sqrt((hx2-x)^2+(hy2-y)^2) or LIQUID_LEN

        out[i+1] = math.max(0.0, (d2 - d1) / LIQUID_LEN)
    end
    return out
end

-- ── 8-sector enemy radar ──────────────────────────────────────────────────
-- Each sector returns normalised distance to nearest enemy (1.0 = none).
local ENEMY_RANGE = 200

local function build_enemy_radar(x, y)
    local sectors = {1,1,1,1,1,1,1,1}
    local ok, enemies = pcall(EntityGetInRadiusWithTag, x, y, ENEMY_RANGE, "enemy")
    if not ok or not enemies then return sectors end
    for _, eid in ipairs(enemies) do
        local tok, ex, ey = pcall(EntityGetTransform, eid)
        if tok then
            local dx, dy = ex - x, ey - y
            local dist   = math.sqrt(dx*dx + dy*dy)
            if dist > 0 then
                local norm   = math.min(dist / ENEMY_RANGE, 1.0)
                local angle  = math.atan2(dy, dx)                     -- -π..π
                local sector = math.floor((angle + math.pi) / (2*math.pi) * 8 + 0.5) % 8 + 1
                if norm < sectors[sector] then sectors[sector] = norm end
            end
        end
    end
    return sectors
end

-- ── 8-sector projectile radar ────────────────────────────────────────────
-- Each sector returns normalised distance to nearest incoming projectile (1.0 = none).
local PROJECTILE_RANGE = 150

local function build_projectile_radar(x, y)
    local sectors = {1,1,1,1,1,1,1,1}
    local ok, projs = pcall(EntityGetInRadiusWithTag, x, y, PROJECTILE_RANGE, "projectile")
    if not ok or not projs then return sectors end
    for _, pid in ipairs(projs) do
        local tok, px, py = pcall(EntityGetTransform, pid)
        if tok then
            local dx, dy = px - x, py - y
            local dist   = math.sqrt(dx*dx + dy*dy)
            if dist > 0 then
                local norm   = math.min(dist / PROJECTILE_RANGE, 1.0)
                local angle  = math.atan2(dy, dx)
                local sector = math.floor((angle + math.pi) / (2*math.pi) * 8 + 0.5) % 8 + 1
                if norm < sectors[sector] then sectors[sector] = norm end
            end
        end
    end
    return sectors
end

-- ── 8-sector gold/loot radar ─────────────────────────────────────────────
-- Signal: 1=no gold nearby, 0=gold at player position.
local GOLD_RANGE = 150

local function build_gold_radar(x, y)
    local sectors = {1,1,1,1,1,1,1,1}
    local ok, nuggets = pcall(EntityGetInRadiusWithTag, x, y, GOLD_RANGE, "gold_nugget")
    if not ok or not nuggets then return sectors end
    for _, gid in ipairs(nuggets) do
        local tok, gx, gy = pcall(EntityGetTransform, gid)
        if tok then
            local dx, dy = gx - x, gy - y
            local dist   = math.sqrt(dx*dx + dy*dy)
            if dist > 0 then
                local norm   = math.min(dist / GOLD_RANGE, 1.0)
                local angle  = math.atan2(dy, dx)
                local sector = math.floor((angle + math.pi) / (2*math.pi) * 8 + 0.5) % 8 + 1
                if norm < sectors[sector] then sectors[sector] = norm end
            end
        end
    end
    return sectors
end

-- ── Portal signal (Holy Mountain teleporter) ─────────────────────────────
-- Returns {dist_norm, dx_norm, dy_norm} in [0,1]:
--   dist_norm = 1   → no portal within PORTAL_RANGE
--   dx_norm   = 0.5 → portal directly above/below; <0.5 left, >0.5 right
--   dy_norm   = 0.5 → portal at same Y; <0.5 above, >0.5 below
-- Holy Mountain teleporters carry tag "teleport_active"; we try a couple of
-- other tags too in case the engine renames them in different biomes.
local PORTAL_RANGE = 400
local PORTAL_TAGS  = { "teleport_active", "portal", "teleportable_NOT_player" }

local function get_portal_signal(x, y)
    local best_d2, best_ex, best_ey = math.huge, nil, nil
    for _, tag in ipairs(PORTAL_TAGS) do
        local ok, ents = pcall(EntityGetWithTag, tag)
        if ok and ents then
            for _, eid in ipairs(ents) do
                local tok, ex, ey = pcall(EntityGetTransform, eid)
                if tok then
                    local d2 = (ex - x)^2 + (ey - y)^2
                    if d2 < best_d2 then best_d2, best_ex, best_ey = d2, ex, ey end
                end
            end
        end
    end
    if not best_ex then return { 1.0, 0.5, 0.5 } end
    local d = math.sqrt(best_d2)
    if d > PORTAL_RANGE then return { 1.0, 0.5, 0.5 } end
    local dx_norm = math.max(0.0, math.min(1.0, (best_ex - x) / PORTAL_RANGE * 0.5 + 0.5))
    local dy_norm = math.max(0.0, math.min(1.0, (best_ey - y) / PORTAL_RANGE * 0.5 + 0.5))
    return { d / PORTAL_RANGE, dx_norm, dy_norm }
end

-- ── Auto-open chests on proximity ────────────────────────────────────────
-- Scans all entities within CHEST_RANGE for ItemChestComponent and kills them.
-- EntityKill() on a chest triggers its death handler: the engine spawns the
-- chest's contents at that position.  We don't need to handle pickup — gold
-- nuggets and items land on the ground and are visible to the existing radars.
--
-- Tag "chest" is not in the API docs (data.wak is packed), so we use the
-- slower but reliable approach: check every entity in radius for the component.
-- At CHEST_RANGE = 22px the entity count is small (typically 0–3).
local function auto_open_chests(player, x, y)
    local ok, ents = pcall(EntityGetInRadius, x, y, CHEST_RANGE)
    if not ok or not ents then return 0 end
    local opened = 0
    for _, eid in ipairs(ents) do
        if eid ~= player then
            local chest_comp = EntityGetFirstComponent(eid, "ItemChestComponent")
            if chest_comp then
                local cok, cx, cy = pcall(EntityGetTransform, eid)
                if cok then
                    info(string.format(
                        "Chest opened at (%.0f, %.0f) ep=%d step=%d",
                        cx, cy, episode_num, episode_steps))
                end
                pcall(EntityKill, eid)
                opened = opened + 1
                chests_opened_total = chests_opened_total + 1
                chests_opened_ep    = chests_opened_ep    + 1
            end
        end
    end
    return opened
end

-- ── Jetpack fuel (0=empty, 1=full) ───────────────────────────────────────
local JETPACK_MAX = 1000.0   -- default mFlyingTimeLeft value = full tank

local function get_jetpack_fuel(player)
    local cdata = EntityGetFirstComponent(player, "CharacterDataComponent")
    if not cdata then return 1.0 end
    local left = cget(cdata, "mFlyingTimeLeft") or JETPACK_MAX
    return math.max(0.0, math.min(1.0, left / JETPACK_MAX))
end

-- ── Wand ready (1=can fire, 0=on cooldown) ───────────────────────────────
-- The player has many children (perks, inventory items, the wand). We want the
-- WAND specifically — wand entities have tag "wand" or "card_action". Reading
-- AbilityComponent from the first child (e.g. a perk) gives garbage cooldowns.
local function get_wand_ready(player)
    local frame    = GameGetFrameNum()
    local children = EntityGetAllChildren(player) or {}
    local wand_ab  = nil
    for _, child in ipairs(children) do
        local is_wand = pcall(EntityHasTag, child, "wand") and EntityHasTag(child, "wand")
        if is_wand then
            wand_ab = EntityGetFirstComponent(child, "AbilityComponent")
            if wand_ab then break end
        end
    end
    if not wand_ab then
        -- Fall back: any AbilityComponent on any child (old behaviour).
        for _, child in ipairs(children) do
            local ab = EntityGetFirstComponent(child, "AbilityComponent")
            if ab then wand_ab = ab; break end
        end
    end
    if not wand_ab then return 1.0 end
    local next_use = cget(wand_ab, "mNextFrameUsable") or 0
    return (frame >= next_use) and 1.0 or 0.0
end

-- ── Build 16-ray state table ──────────────────────────────────────────────
local function build_rays(x, y)
    local rays = {}
    for i = 0, 15 do
        local angle = i * (math.pi / 8)
        local ok, hit, hx, hy = pcall(
            RaytracePlatforms,
            x, y,
            x + math.cos(angle) * 150,
            y + math.sin(angle) * 150
        )
        if ok and hit then
            rays[i+1] = math.sqrt((hx-x)^2 + (hy-y)^2) / 150
        else
            rays[i+1] = 1.0
        end
    end
    return rays
end

-- ── Prevent auto-pause on focus loss during RL training ──────────────────
function OnPausedChanged(is_paused, is_inventory_pause)
    if is_paused and not is_inventory_pause then
        GameSetPaused(false)
    end
end

-- ── Pre-update: apply buffered action BEFORE physics ─────────────────────
function OnWorldPreUpdate()
    if not RaytracePlatforms then return end
    local player = EntityGetWithTag("player_unit")[1]
    if player then apply_action(player, pending_action) end
end

-- ── Post-update: gather state, HUD, communicate with Python ──────────────
function OnWorldPostUpdate()
    if not RaytracePlatforms then return end

    local frame  = GameGetFrameNum()
    local player = EntityGetWithTag("player_unit")[1]

    -- Simple FPS estimate for diagnostic logging
    local now = os.clock()
    table.insert(frame_times, now)
    if #frame_times > PERF_WINDOW then table.remove(frame_times, 1) end
    local fps = 0
    if #frame_times >= 2 then
        fps = (#frame_times - 1) / (frame_times[#frame_times] - frame_times[1])
    end

    -- HUD ──────────────────────────────────────────────────────────────────
    if not gui then gui = GuiCreate() end
    GuiStartFrame(gui)
    GuiIdPushString(gui, "rl_hud")

    if not socket then
        local retry = math.max(0, connection_retry_interval - (frame - last_connection_attempt))
        GuiColorSetForNextWidget(gui, 1, 0.4, 0.2, 1)
        GuiText(gui, 10, 10, string.format("RL AGENT  DISCONNECTED  retry:%d  fps:%.0f", retry, fps))
    else
        local act = ACTION_NAMES[pending_action] or "?"
        GuiColorSetForNextWidget(gui, 0.4, 1, 0.4, 1)
        GuiText(gui, 10, 10, string.format(
            "RL AGENT  Ep:%-3d  Step:%-5d  %-5s  fps:%.0f",
            episode_num, episode_steps, act, fps))
    end
    GuiColorSetForNextWidget(gui, 1, 1, 1, 1)
    GuiIdPop(gui)

    -- Connection management ────────────────────────────────────────────────
    if not socket then
        if frame - last_connection_attempt > connection_retry_interval then
            info("Connecting → " .. WS_URL)
            last_connection_attempt = frame
            consecutive_errors = 0
            local ok, s = pcall(pollnet.open_ws, WS_URL)
            if ok then socket = s else err("open_ws failed: " .. tostring(s)) end
        end
        return
    end

    -- Poll socket ──────────────────────────────────────────────────────────
    local poll_ok, happy, msg = pcall(socket.poll, socket)
    if not poll_ok then
        err("socket:poll() threw: " .. tostring(happy))
        socket = nil; return
    end

    local st = socket:status()

    if st == "error" then
        consecutive_errors = consecutive_errors + 1
        local emsg = ""
        pcall(function()
            local buf = ffi and ffi.new("char[512]") or nil
            if buf then emsg = ffi.string(buf) end
        end)
        warn(string.format("Socket error #%d — will reconnect", consecutive_errors))
        socket = nil; pending_action = 0; return
    end

    if st == "closed" then
        info("Socket closed — will reconnect")
        socket = nil; pending_action = 0; return
    end

    -- Buffer incoming action ───────────────────────────────────────────────
    if msg and type(msg) == "string" and #msg > 0 then
        local ok2, act = pcall(json.decode, msg)
        if ok2 and type(act) == "number" then
            pending_action = math.floor(act)
        else
            warn("Bad action payload: " .. msg:sub(1, 40))
        end
    end

    if st ~= "open" then return end
    if not player then return end

    -- Frame skip: build/send state only every FRAME_SKIP frames ───────────
    if (frame % FRAME_SKIP) ~= 0 then return end

    -- Player state ─────────────────────────────────────────────────────────
    local x, y
    do
        local ok3, ex, ey = pcall(EntityGetTransform, player)
        if not ok3 then warn("EntityGetTransform failed"); return end
        x, y = ex, ey
    end

    local cdata = EntityGetFirstComponent(player, "CharacterDataComponent")

    if not spawn_x then
        -- First frame: drop the player from the surface into the Mines.
        -- The surface biome is open terrain with no corridors — the agent
        -- wanders aimlessly there. We raycast straight down to find the
        -- first platform and teleport just above it.
        local target_x, target_y = x, y + 400  -- fallback if raycast misses
        local rok, hit, _hx, hy = pcall(
            RaytracePlatforms, x, y + 10, x, y + INITIAL_DESCENT_RANGE
        )
        if rok and hit then
            target_y = hy - INITIAL_DESCENT_LIFT
        end

        pcall(EntitySetTransform, player, target_x, target_y)
        if cdata then cset(cdata, "mVelocity", 0, 0) end

        spawn_x, spawn_y = target_x, target_y
        spawn_candidates[#spawn_candidates + 1] = { x = target_x, y = target_y }
        initial_descent_done = true
        episode_num = 1
        info(string.format(
            "Spawn recorded (%.0f, %.0f) — descended from surface (%.0f, %.0f)",
            target_x, target_y, x, y))

        -- Reflect the teleport in this frame's state so observation is correct
        x, y = target_x, target_y
    end

    -- Virtual-HP system: keep engine HP at IMMORTAL_HP so Noita never kills
    -- the player entity; track damage in virtual_hp ourselves.
    local dmg = EntityGetFirstComponent(player, "DamageModelComponent")
    if dmg then
        cset(dmg, "max_hp", IMMORTAL_HP)
        local engine_hp = cget(dmg, "hp") or IMMORTAL_HP
        if engine_hp < IMMORTAL_HP then
            virtual_hp = virtual_hp - (IMMORTAL_HP - engine_hp)
            cset(dmg, "hp", IMMORTAL_HP)
        end
    end
    local hp_norm = math.max(0.0, virtual_hp / VIRTUAL_MAX_HP)

    local vx, vy, on_ground = 0.0, 0.0, false
    if cdata then
        vx, vy    = cget(cdata, "mVelocity")
        on_ground = cget(cdata, "is_on_ground")
        vx = vx or 0; vy = vy or 0; on_ground = on_ground or false
    end

    local rays             = build_rays(x, y)
    local liquid_sensors   = build_liquid_sensors(x, y)
    local enemy_radar      = build_enemy_radar(x, y)
    local projectile_radar = build_projectile_radar(x, y)
    local gold_radar       = build_gold_radar(x, y)
    local portal_signal    = get_portal_signal(x, y)
    local jetpack_fuel     = get_jetpack_fuel(player)
    local wand_ready       = get_wand_ready(player)

    -- Fire: read DamageModelComponent.is_on_fire (the engine sets this directly).
    -- The GameEffect "ON_FIRE" doesn't exist in Noita — burning is a stain/material
    -- contact, not a game effect, so GameGetGameEffectCount always returned 0.
    local is_on_fire = 0.0
    if dmg then
        is_on_fire = (cget(dmg, "is_on_fire") == true) and 1.0 or 0.0
    end
    -- Poison/radiation: only RADIOACTIVE is a valid enum in this build. Other names
    -- (POISONED / STAINED_RADIOACTIVE) make the C++ enum parser spam stderr — those
    -- warnings bypass Lua pcall, so we MUST stick to the one valid name here.
    local is_poisoned = 0.0
    local ok_eff, fc = pcall(GameGetGameEffectCount, player, "RADIOACTIVE")
    if ok_eff and fc and fc > 0 then is_poisoned = 1.0 end

    -- Sky visibility: 1=open sky, 0=deep underground (depth proxy)
    local sky_ok, sky_v = pcall(GameGetSkyVisibility, x, y)
    local sky_visibility = (sky_ok and sky_v) and math.max(0.0, math.min(1.0, sky_v)) or 0.0

    -- Auto-open nearby chests BEFORE building state (so the frame that triggers
    -- opening is reported with the updated chests_opened_ep counter).
    -- Only run on action frames (frame % FRAME_SKIP == 0) to keep cost low.
    auto_open_chests(player, x, y)

    -- Current gold and kill count for reward tracking in Python
    local gold = 0
    local wallet = EntityGetFirstComponent(player, "WalletComponent")
    if wallet then gold = cget(wallet, "money") or 0 end

    local ok_st, k_str = pcall(StatsGetValue, "enemies_killed")
    local kills = (ok_st and k_str) and tonumber(k_str) or 0

    -- Draw radar ───────────────────────────────────────────────────────────
    GuiIdPushString(gui, "rl_radar")
    draw_radar(gui, 10, 24, rays)
    GuiIdPop(gui)

    episode_steps = episode_steps + 1

    -- Periodic diagnostic log ──────────────────────────────────────────────
    if (frame % 300) == 0 then
        info(string.format(
            "DIAG ep=%d step=%d pos=(%.0f,%.0f) vel=(%.1f,%.1f) vhp=%.2f gnd=%s act=%s chests=%d fps=%.0f",
            episode_num, episode_steps, x, y, vx, vy, virtual_hp,
            tostring(on_ground), ACTION_NAMES[last_action] or "?",
            chests_opened_ep, fps))
    end

    -- Death detection: virtual HP exhausted OR episode timeout ──────────────
    if virtual_hp <= 0.0 or episode_steps >= MAX_EP_STEPS then
        local reason = (virtual_hp <= 0.0) and "DEAD" or "TIMEOUT"
        info(string.format("Ep %d ended  steps=%d  reason=%s  pos=(%.0f,%.0f)  vhp=%.3f",
            episode_num, episode_steps, reason, x, y, virtual_hp))
        local dead_state = {
            x=x, y=y, hp=0.0, vx=0.0, vy=0.0,
            rays=rays, liquid_sensors=liquid_sensors, enemy_radar=enemy_radar,
            projectile_radar=projectile_radar, gold_radar=gold_radar,
            portal=portal_signal,
            jetpack_fuel=1.0, wand_ready=1.0,
            is_on_fire=0.0, is_poisoned=0.0, sky_visibility=sky_visibility,
            gold=gold, kills=kills, chests=chests_opened_ep,
            dead=true, on_ground=false, frame=frame
        }
        local ok4, encoded = pcall(json.encode, dead_state)
        if ok4 then pcall(socket.send, socket, encoded) end
        respawn_player(player)
        return
    end

    -- Send live state ──────────────────────────────────────────────────────
    local state = {
        x=x, y=y, hp=hp_norm, vx=vx, vy=vy,
        rays=rays, liquid_sensors=liquid_sensors, enemy_radar=enemy_radar,
        projectile_radar=projectile_radar, gold_radar=gold_radar,
        portal=portal_signal,
        jetpack_fuel=jetpack_fuel, wand_ready=wand_ready,
        is_on_fire=is_on_fire, is_poisoned=is_poisoned,
        sky_visibility=sky_visibility, gold=gold, kills=kills,
        chests=chests_opened_ep,
        dead=false, on_ground=on_ground, frame=frame
    }
    local ok5, encoded = pcall(json.encode, state)
    if ok5 then
        local ok6, send_err = pcall(socket.send, socket, encoded)
        if not ok6 then
            warn("socket:send failed: " .. tostring(send_err))
            consecutive_errors = consecutive_errors + 1
            if consecutive_errors >= MAX_ERRORS then
                err("Too many send errors — resetting socket")
                socket = nil; pending_action = 0
            end
        else
            consecutive_errors = 0
        end
    else
        warn("json.encode failed: " .. tostring(encoded))
    end
end
