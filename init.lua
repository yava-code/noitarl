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
info("Mod init started — noitarl v0.2")

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
local FRAME_SKIP     = 4     -- accept new action / send state every N frames

-- Immortal-agent HP hack: engine never kills the player;
-- we track "virtual HP" ourselves and teleport-respawn when it hits 0.
local IMMORTAL_HP    = 10000.0
local VIRTUAL_MAX_HP = 4.0    -- 4.0 engine units ≈ 100% HP in UI
local virtual_hp     = VIRTUAL_MAX_HP

-- ── Per-frame state ───────────────────────────────────────────────────────
local pending_action  = 0
local last_action     = 0
local ACTION_NAMES    = {[0]="IDLE",[1]="LEFT",[2]="RIGHT",[3]="JUMP",[4]="FIRE"}
local FORCE_RESPAWN   = 99   -- special signal from Python to force episode reset

-- ── Episode tracking ──────────────────────────────────────────────────────
local spawn_x, spawn_y = 400.0, 50.0  -- entrance to the first mines
local episode_num      = 0
local episode_steps    = 0
local frame_times      = {}   -- rolling window for FPS estimate
local PERF_WINDOW      = 60

-- ── Apply action: smoothed velocity + manual jetpack ─────────────────────
local function apply_action(player, action)
    if not player or player == 0 then return end

    -- Special signal: Python ended the episode via step-limit; teleport back to spawn
    if action == FORCE_RESPAWN then
        respawn_player(player)
        pending_action = 0
        return
    end

    last_action = action

    local cdata = EntityGetFirstComponent(player, "CharacterDataComponent")
    if cdata then
        local vx, vy    = cget(cdata, "mVelocity")
        local on_ground = cget(cdata, "is_on_ground")
        local fuel      = cget(cdata, "mFlyingTimeLeft") or 1000
        vx = vx or 0; vy = vy or 0

        -- Horizontal: lerp toward target — ground has more grip than air
        local target_vx = 0
        if     action == 1 then target_vx = -MOVE_SPEED
        elseif action == 2 then target_vx =  MOVE_SPEED
        end
        local grip = on_ground and 0.3 or 0.05
        vx = vx + (target_vx - vx) * grip

        -- Jump / jetpack
        if action == 3 then
            if on_ground then
                vy = JUMP_SPEED
            elseif fuel > 0 then
                vy   = math.max(vy - 12, -200)
                fuel = math.max(0, fuel - 20)
            end
        end

        cset(cdata, "mFlyingTimeLeft", fuel)
        cset(cdata, "mVelocity", vx, vy)
    end

    -- Fire + auto-aim on ControlsComponent
    local ctrl = EntityGetFirstComponent(player, "ControlsComponent")
    if ctrl then
        -- Auto-aim: track nearest enemy every frame
        local tok, px, py = pcall(EntityGetTransform, player)
        if tok then
            local aok, enemies = pcall(EntityGetInRadiusWithTag, px, py, 250, "enemy")
            local nx, ny = px + 50, py   -- default: face right
            if aok and enemies and #enemies > 0 then
                local nearest_d2 = math.huge
                for _, eid in ipairs(enemies) do
                    local eok, ex, ey = pcall(EntityGetTransform, eid)
                    if eok then
                        local d2 = (ex-px)^2 + ((ey+4)-py)^2
                        if d2 < nearest_d2 then nearest_d2, nx, ny = d2, ex, ey+4 end
                    end
                end
            end
            local dx, dy = nx - px, ny - py
            local len = math.sqrt(dx*dx + dy*dy)
            if len > 0.001 then
                cset(ctrl, "mAimingVectorNormalized", dx/len, dy/len)
                cset(ctrl, "mMousePosition", nx, ny)   -- vec2 field, two floats
            end
        end

        -- Fire: only update mButtonFrameFire on the rising edge so cast_delay isn't
        -- reset every frame (which would prevent any projectile from actually launching).
        local fire     = (action == 4)
        local was_fire = (cget(ctrl, "mButtonDownFire") == true)

        cset(ctrl, "mButtonDownFire",      fire)
        cset(ctrl, "mButtonDownLeftClick", fire)

        if fire and not was_fire then
            local frame = GameGetFrameNum()
            cset(ctrl, "mButtonFrameFire",      frame)
            cset(ctrl, "mButtonFrameLeftClick", frame)
        end
    end
end

-- ── Respawn: teleport to spawn, reset virtual HP, flush status effects ───
local function respawn_player(player)
    virtual_hp = VIRTUAL_MAX_HP

    if spawn_x and spawn_y then
        local ok, e = pcall(EntitySetTransform, player, spawn_x, spawn_y)
        if not ok then warn("EntitySetTransform failed: " .. tostring(e)) end
    end

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

    episode_num    = episode_num + 1
    episode_steps  = 0
    pending_action = 0
    info(string.format("Ep %d started — spawn (%.0f, %.0f)", episode_num, spawn_x or 0, spawn_y or 0))
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
    local frame    = GameGetFrameNum()
    local children = EntityGetAllChildren(player) or {}
    for _, child in ipairs(children) do
        local ab = EntityGetFirstComponent(child, "AbilityComponent")
        if ab then
            local next_use = cget(ab, "mNextFrameUsable") or 0
            return (frame >= next_use) and 1.0 or 0.0
        end
    end
    return 1.0
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

    if not spawn_x then
        spawn_x, spawn_y = x, y
        episode_num = 1
        info(string.format("Spawn recorded (%.0f, %.0f)", x, y))
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
    local cdata = EntityGetFirstComponent(player, "CharacterDataComponent")
    if cdata then
        vx, vy   = cget(cdata, "mVelocity")
        on_ground = cget(cdata, "is_on_ground")
        vx = vx or 0; vy = vy or 0; on_ground = on_ground or false
    end

    local rays             = build_rays(x, y)
    local liquid_sensors   = build_liquid_sensors(x, y)
    local enemy_radar      = build_enemy_radar(x, y)
    local projectile_radar = build_projectile_radar(x, y)
    local gold_radar       = build_gold_radar(x, y)
    local jetpack_fuel     = get_jetpack_fuel(player)
    local wand_ready       = get_wand_ready(player)

    local ok1, fc1 = pcall(GameGetGameEffectCount, player, "ON_FIRE")
    local is_on_fire  = (ok1 and fc1 and fc1 > 0) and 1.0 or 0.0
    local ok2, fc2 = pcall(GameGetGameEffectCount, player, "RADIOACTIVE")
    local is_poisoned = (ok2 and fc2 and fc2 > 0) and 1.0 or 0.0

    -- Sky visibility: 1=open sky, 0=deep underground (depth proxy)
    local sky_ok, sky_v = pcall(GameGetSkyVisibility, x, y)
    local sky_visibility = (sky_ok and sky_v) and math.max(0.0, math.min(1.0, sky_v)) or 0.0

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
            "DIAG ep=%d step=%d pos=(%.0f,%.0f) vel=(%.1f,%.1f) vhp=%.2f gnd=%s act=%s fps=%.0f",
            episode_num, episode_steps, x, y, vx, vy, virtual_hp,
            tostring(on_ground), ACTION_NAMES[last_action] or "?", fps))
    end

    -- Death detection (virtual HP exhausted) ─────────────────────────────
    if virtual_hp <= 0.0 then
        info(string.format("Ep %d ended  steps=%d  pos=(%.0f,%.0f)  vhp=%.3f",
            episode_num, episode_steps, x, y, virtual_hp))
        local dead_state = {
            x=x, y=y, hp=0.0, vx=0.0, vy=0.0,
            rays=rays, liquid_sensors=liquid_sensors, enemy_radar=enemy_radar,
            projectile_radar=projectile_radar, gold_radar=gold_radar,
            jetpack_fuel=1.0, wand_ready=1.0,
            is_on_fire=0.0, is_poisoned=0.0, sky_visibility=sky_visibility,
            gold=gold, kills=kills,
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
        jetpack_fuel=jetpack_fuel, wand_ready=wand_ready,
        is_on_fire=is_on_fire, is_poisoned=is_poisoned,
        sky_visibility=sky_visibility, gold=gold, kills=kills,
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
