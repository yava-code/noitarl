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

-- ── Optional game-speed multiplier (noita_dev.exe only) ───────────────────
-- Read from speed.txt next to this mod. Values > 1 run the simulation faster.
-- 2.0 is stable; > 3.0 may cause physics glitches.
-- Silently skipped on the release build where SetGameSpeed doesn't exist.
local function read_game_speed()
    local f = io.open(mod_path("speed.txt"), "r")
    if f then
        local v = tonumber(f:read("*l"))
        f:close()
        if v and v > 0 then return v end
    end
    return nil  -- nil = don't touch game speed
end
local GAME_SPEED = read_game_speed()
if GAME_SPEED then
    local ok, err2 = pcall(SetGameSpeed, GAME_SPEED)
    if ok then
        info(string.format("Game speed set to %.1fx", GAME_SPEED))
    else
        warn("SetGameSpeed not available (release build?): " .. tostring(err2))
    end
end

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

-- ── Last-frame cached state for HUD (updated each action frame) ──────────
local hud_vhp      = VIRTUAL_MAX_HP
local hud_sky      = 0.0
local hud_portal   = 1.0   -- portal[1] = dist_norm
local hud_probs    = {}
local hud_saliency = {}

-- ── Per-frame state ───────────────────────────────────────────────────────
-- Discrete 18 action space:
--   0 IDLE | 1 L | 2 R | 3 UP | 4 UL | 5 UR
--   6 JETPK | 7 KICK
--   8 F_R | 9 F_UR | 10 F_U | 11 F_UL
--   12 F_L | 13 F_DL | 14 F_D | 15 F_DR
--   16 F_AUTO | 17 F_SMART
local pending_action   = 0
local last_action      = 0
local last_facing      = 1     -- last non-zero horizontal direction (for KICK aim)
local last_kick_frame  = -1000
local agent_was_firing = false -- Track fire state locally so engine hardware poll doesn't break it
-- Camera-jitter fix: only rewrite ControlsComponent aim when it actually changes.
-- Noita's camera follows the aim cursor; rewriting it every action-frame caused
-- visible "convulsions" 3-5x/sec. Track last applied aim so we can skip no-op writes.
local last_aim_type     = -1
local last_aim_target_x = nil
local last_aim_target_y = nil
local ACTION_NAMES    = {
    [0]="IDLE", [1]="L", [2]="R", [3]="UP", [4]="UL", [5]="UR",
    [6]="JETPK", [7]="KICK",
    [8]="F_R", [9]="F_UR", [10]="F_U", [11]="F_UL",
    [12]="F_L", [13]="F_DL", [14]="F_D", [15]="F_DR",
    [16]="F_AUTO", [17]="F_SMART"
}
-- {move_x, do_jump, do_fire, aim_type, aim_x, aim_y, do_kick, do_jetpack}
-- aim_type: 0=move_dir, 1=fixed_vec, 2=auto_enemy, 3=smart_loot
local ACTION_DECODE = {
    [0] = { 0, false, false, 0, 0, 0, false, false },
    [1] = {-1, false, false, 0, 0, 0, false, false },
    [2] = { 1, false, false, 0, 0, 0, false, false },
    [3] = { 0, true,  false, 0, 0, 0, false, false },
    [4] = {-1, true,  false, 0, 0, 0, false, false },
    [5] = { 1, true,  false, 0, 0, 0, false, false },
    [6] = { 0, false, false, 0, 0, 0, false, true  },
    [7] = { 0, false, false, 0, 0, 0, true,  false },
    [8] = { 0, false, true,  1,  1,  0, false, false }, -- Fire Right
    [9] = { 0, false, true,  1,  1, -1, false, false }, -- Fire Up-Right
    [10]= { 0, false, true,  1,  0, -1, false, false }, -- Fire Up
    [11]= { 0, false, true,  1, -1, -1, false, false }, -- Fire Up-Left
    [12]= { 0, false, true,  1, -1,  0, false, false }, -- Fire Left
    [13]= { 0, false, true,  1, -1,  1, false, false }, -- Fire Down-Left
    [14]= { 0, false, true,  1,  0,  1, false, false }, -- Fire Down
    [15]= { 0, false, true,  1,  1,  1, false, false }, -- Fire Down-Right
    [16]= { 0, false, true,  2,  0,  0, false, false }, -- Auto-aim enemy
    [17]= { 0, false, true,  3,  0,  0, false, false }, -- Smart-aim loot
}
local MAX_EP_STEPS    = 2000  -- ~2.2 min at 60 fps with FRAME_SKIP=4

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
    local move_x, do_jump, do_fire, aim_type, aim_x, aim_y, do_kick, do_jetpack =
        decode[1], decode[2], decode[3], decode[4], decode[5], decode[6], decode[7], decode[8]

    if move_x ~= 0 then last_facing = move_x end

    -- Physics: write velocity directly (mVelocity bypasses Noita's input layer)
    local vx_out, vy_out, on_ground_now = 0, 0, false
    local cdata = EntityGetFirstComponent(player, "CharacterDataComponent")
    if cdata then
        local cur_vx, cur_vy = cget(cdata, "mVelocity")
        local on_ground      = cget(cdata, "is_on_ground")
        cur_vx = cur_vx or 0; cur_vy = cur_vy or 0
        on_ground_now = on_ground == true

        local target_vx = move_x * MOVE_SPEED
        local new_vx    = target_vx
        local new_vy    = cur_vy
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

    -- KICK: melee damage to enemies within KICK_RANGE
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
                            if (ex - px) * last_facing >= -8 then
                                pcall(EntityInflictDamage, eid, KICK_DAMAGE, "SLICE",
                                    "kicked", "RAGDOLL_SOFT", last_facing * 250, -50, player,
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
            local nx, ny = px + 50 * last_facing, py -- default: horizontal
            
            if aim_type == 1 then
                -- Fixed vector (8 directions)
                nx, ny = px + aim_x * 50, py + aim_y * 50
            elseif aim_type == 2 then
                -- Auto-aim: nearest enemy in 250 px
                local aok, enemies = pcall(EntityGetInRadiusWithTag, px, py, 250, "enemy")
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
            elseif aim_type == 3 then
                -- Smart-aim: nearest loot (chest or nugget)
                local best_d2, bx, by = math.huge, nil, nil
                local loot_tags = {"gold_nugget", "chest", "item_pickup"}
                for _, tag in ipairs(loot_tags) do
                    local lok, ents = pcall(EntityGetInRadiusWithTag, px, py, 200, tag)
                    if lok and ents then
                        for _, eid in ipairs(ents) do
                            local eok, ex, ey = pcall(EntityGetTransform, eid)
                            if eok then
                                local d2 = (ex-px)^2 + (ey-py)^2
                                if d2 < best_d2 then best_d2, bx, by = d2, ex, ey end
                            end
                        end
                    end
                end
                if bx then nx, ny = bx, by end
            end

            local dx, dy = nx - px, ny - py
            local len = math.sqrt(dx*dx + dy*dy)
            if len > 0.001 then
                local nxv, nyv = dx/len, dy/len

                -- Camera-jitter fix: only rewrite the aim cursor when the agent is
                -- actually firing, the aim mode changed, or an enemy/loot target
                -- moved far enough to matter. For pure-movement actions (aim_type==0)
                -- we leave the cursor where the engine last had it.
                local target_moved =
                    last_aim_target_x and
                    (math.abs(nx - last_aim_target_x) > 30
                     or math.abs(ny - last_aim_target_y) > 30)
                local should_update_aim =
                    do_fire
                    or aim_type ~= last_aim_type
                    or (aim_type >= 2 and target_moved)

                if aim_type ~= 0 and should_update_aim then
                    cset(ctrl, "mAimingVectorNormalized", nxv, nyv)
                    cset(ctrl, "mAimingVector",           nxv * 40, nyv * 40)
                    cset(ctrl, "mMousePosition",          nx, ny)
                    cset(ctrl, "mMousePositionRaw",       nx, ny)
                    cset(ctrl, "mGamePadCursorInWorld",   nx, ny)
                    cset(ctrl, "mGamepadIndirectAiming",  nxv, nyv)
                    -- Intentionally NOT touching mSmoothedAimingVector — leave
                    -- engine interpolation alone so the camera glides instead of snapping.
                    last_aim_type, last_aim_target_x, last_aim_target_y = aim_type, nx, ny
                end

                -- Facing always tracks intent so KICK / sprites stay correct, but
                -- this single int doesn't move the camera.
                local cplat = EntityGetFirstComponent(player, "CharacterPlatformingComponent")
                if cplat then cset(cplat, "mFacingDirection", (nxv >= 0) and 1 or -1) end
            end
        end

        -- Fire: Fix "eternal first frame" bug by using local agent_was_firing
        cset(ctrl, "mButtonDownFire",      do_fire)
        cset(ctrl, "mButtonDownLeftClick", do_fire)
        
        if do_fire and not agent_was_firing then
            local frame = GameGetFrameNum()
            cset(ctrl, "mButtonFrameFire",      frame)
            cset(ctrl, "mButtonFrameLeftClick", frame)
            agent_was_firing = true
        elseif not do_fire then
            agent_was_firing = false
        end
    end

    if cdata then
        log_action_trace({
            f  = GameGetFrameNum(), a  = action,
            mx = move_x, jp = do_jump and 1 or 0,
            fr = do_fire and 1 or 0, at = aim_type,
            vx = vx_out, vy = vy_out, gnd = on_ground_now and 1 or 0,
        })
    end
end

-- ── Respawn: teleport to (possibly randomised) spawn, reset state ────────
local function pick_spawn()
    -- Prefer the deepest recorded spawn points (higher Y = deeper in Noita).
    -- Picking from the top-3 deepest candidates prevents the agent from
    -- repeatedly respawning in narrow surface-adjacent corridors from early
    -- episodes, while still keeping some variety.
    local n = #spawn_candidates
    if n == 0 then return spawn_x or 0.0, spawn_y or 0.0 end
    local sorted = {}
    for i = 1, n do sorted[i] = spawn_candidates[i] end
    table.sort(sorted, function(a, b) return a.y > b.y end)
    local sp = sorted[math.random(math.min(3, n))]
    local jx = math.random(-SPAWN_JITTER, SPAWN_JITTER)
    return sp.x + jx, sp.y
end

local function respawn_player(player)
    virtual_hp = VIRTUAL_MAX_HP

    -- Pick a spawn from the underground candidate pool (populated on first
    -- descent). If the pool is somehow empty, redo the INITIAL_DESCENT_RANGE
    -- raycast from the player's current position so we always land in the Mines.
    local sx, sy
    if #spawn_candidates > 0 then
        sx, sy = pick_spawn()
    else
        local ok0, px, py = pcall(EntityGetTransform, player)
        sx, sy = ok0 and px or 0, ok0 and py or 0
        local rok, hit, _hx, hy = pcall(
            RaytracePlatforms, sx, sy + 10, sx, sy + INITIAL_DESCENT_RANGE)
        if rok and hit then
            sy = hy - INITIAL_DESCENT_LIFT
        else
            sy = sy + 400  -- fallback if raycast misses
        end
        spawn_candidates[#spawn_candidates + 1] = { x = sx, y = sy }
        warn("spawn_candidates was empty — re-descended to (" .. sx .. "," .. sy .. ")")
    end

    local ok, e = pcall(EntitySetTransform, player, sx, sy)
    if not ok then warn("EntitySetTransform failed: " .. tostring(e)) end

    local cdata = EntityGetFirstComponent(player, "CharacterDataComponent")
    if cdata then
        cset(cdata, "mVelocity", 0, 0)
        cset(cdata, "mFlyingTimeLeft", 1000.0)
    end

    -- Clear status effects
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

local function draw_radar(g, ox, oy, rays, saliency)
    local cell = 8
    local s_threshold = 0.05
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
                local s = saliency and saliency[idx] or 0.0
                
                if s > s_threshold then
                    -- Highly salient: white border/highlight
                    GuiColorSetForNextWidget(g, 1, 1, 1, 1)
                else
                    GuiColorSetForNextWidget(g, 1-d, d, 0, 1)
                end
                GuiText(g, tx, ty, dist_char(d))
            end
        end
    end
    GuiColorSetForNextWidget(g, 1, 1, 1, 1)
end

local function draw_probs(g, ox, oy, probs)
    local bar_max_w = 40
    for i = 0, 9 do
        local p = probs[i+1] or 0.0
        local name = ACTION_NAMES[i] or "?"
        local ty = oy + i * 10
        
        -- Draw bar background
        GuiColorSetForNextWidget(g, 0.2, 0.2, 0.2, 1)
        GuiText(g, ox + 35, ty, "....................")
        
        -- Draw bar
        local bar_len = math.floor(p * 20)
        if bar_len > 0 then
            GuiColorSetForNextWidget(g, 0.4, 0.8, 1, 1)
            GuiText(g, ox + 35, ty, string.rep("|", bar_len))
        end
        
        -- Draw text
        GuiColorSetForNextWidget(g, 0.8, 0.8, 0.8, 1)
        GuiText(g, ox, ty, string.format("%-5s %3d%%", name, math.floor(p*100)))
    end
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
local PORTAL_RANGE = 600   -- increased from 400 so HM exit portal is visible from entry side
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
local function get_wand_ready(player)
    local frame = GameGetFrameNum()
    local inv2  = EntityGetFirstComponent(player, "Inventory2Component")
    if not inv2 then return 1.0 end
    
    local active_wand = cget(inv2, "mActiveItem")
    if not active_wand or active_wand == 0 then
        -- Try searching children if Inventory2 fails (unlikely for player_unit)
        local children = EntityGetAllChildren(player) or {}
        for _, child in ipairs(children) do
            if EntityHasTag(child, "wand") then
                active_wand = child
                break
            end
        end
    end

    if not active_wand or active_wand == 0 then return 1.0 end
    
    local wand_ab = EntityGetFirstComponent(active_wand, "AbilityComponent")
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
            "RL AGENT  Ep:%-3d  Step:%-4d/%-4d  %-5s  fps:%.0f",
            episode_num, episode_steps, MAX_EP_STEPS, act, fps))
        -- Second HUD line: internal state for visual debugging (updated from last action frame)
        GuiColorSetForNextWidget(gui, 0.9, 0.9, 0.4, 1)
        GuiText(gui, 10, 20, string.format(
            "  vhp:%.2f  sky:%.2f  portal:%.2f",
            hud_vhp, hud_sky, hud_portal))
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
        local ok2, data = pcall(json.decode, msg)
        if ok2 and type(data) == "table" then
            local act = data.action
            if act == -1 then
                info("Force-respawn requested by Python (cowardice truncation)")
                virtual_hp = 0.0
            elseif act then
                pending_action = math.floor(act)
            end
            hud_probs = data.probs or {}
            hud_saliency = data.saliency or {}
        elseif ok2 and type(data) == "number" then
            -- Backwards compatibility for raw action number
            if data == -1 then
                info("Force-respawn requested by Python (truncation)")
                virtual_hp = 0.0
            else
                pending_action = math.floor(data)
            end
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
        -- We skip the first SKIP_SURFACE_PX of terrain so we don't land on
        -- the surface ledge — we want to be in the actual underground mines.
        local SKIP_SURFACE_PX = 50  -- ignore surface terrain in first 50 px
        local target_x, target_y = x, y + 400  -- fallback: guaranteed underground

        -- Primary: find first platform below the surface layer.
        local rok, hit, _hx, hy = pcall(
            RaytracePlatforms, x, y + SKIP_SURFACE_PX, x, y + INITIAL_DESCENT_RANGE
        )
        if rok and hit then
            target_y = hy - INITIAL_DESCENT_LIFT
        end

        -- Safety: if we ended up suspiciously close to the starting y,
        -- the raycast hit surface terrain — push further down.
        if math.abs(target_y - y) < 100 then
            target_y = y + 400
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

    hud_portal = portal_signal[1]   -- dist_norm: 1=no portal, 0=at portal

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

    -- Update HUD cache (values from this action frame shown next GUI frame)
    hud_vhp    = virtual_hp
    hud_sky    = sky_visibility

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
    draw_radar(gui, 10, 24, rays, hud_saliency)
    GuiIdPop(gui)

    -- Draw probabilities ──────────────────────────────────────────────────
    if #hud_probs > 0 then
        GuiIdPushString(gui, "rl_probs")
        draw_probs(gui, 10, 55, hud_probs)
        GuiIdPop(gui)
    end

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

    -- Camera centre (world coords) — Python side uses this to compute the
    -- player's on-screen pixel position for the CNN crop.
    local cam_ok, cam_x, cam_y = pcall(GameGetCameraPos)
    if not cam_ok then cam_x, cam_y = x, y end
    -- Camera viewport size in world units (covers any zoom changes the player
    -- makes in-game). w/h are equivalent to screen pixels at default zoom.
    local cb_ok, _cbx, _cby, cb_w, cb_h = pcall(GameGetCameraBounds)
    if not cb_ok then cb_w, cb_h = 0, 0 end

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
        dead=false, on_ground=on_ground, frame=frame,
        cam_x=cam_x, cam_y=cam_y, cam_w=cb_w, cam_h=cb_h,
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
