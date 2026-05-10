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
local DEATH_HP       = 0.02
local FRAME_SKIP     = 4     -- accept new action / send state every N frames

-- ── Per-frame state ───────────────────────────────────────────────────────
local pending_action  = 0
local last_action     = 0
local ACTION_NAMES    = {[0]="IDLE",[1]="LEFT",[2]="RIGHT",[3]="JUMP",[4]="FIRE"}

-- ── Episode tracking ──────────────────────────────────────────────────────
local spawn_x, spawn_y = nil, nil
local episode_num      = 0
local episode_steps    = 0
local frame_times      = {}   -- rolling window for FPS estimate
local PERF_WINDOW      = 60

-- ── Apply action: direct velocity write ──────────────────────────────────
-- ControlsComponent.mButtonDown* is overwritten by the engine's keyboard
-- reader before physics; mVelocity is consumed directly by PlayerCollisionSystem.
local function apply_action(player, action)
    if not player or player == 0 then return end
    last_action = action

    local cdata = EntityGetFirstComponent(player, "CharacterDataComponent")
    if cdata then
        local vx, vy    = cget(cdata, "mVelocity")
        local on_ground = cget(cdata, "is_on_ground")
        vx = vx or 0; vy = vy or 0
        if     action == 1 then vx = -MOVE_SPEED
        elseif action == 2 then vx =  MOVE_SPEED
        elseif action == 3 and on_ground then vy = JUMP_SPEED
        end
        cset(cdata, "mVelocity", vx, vy)
    end

    -- Fire via ControlsComponent (button-based, not velocity)
    local ctrl = EntityGetFirstComponent(player, "ControlsComponent")
    if ctrl then
        local fire = (action == 4)
        cset(ctrl, "mButtonDownFire", fire)
        if fire then cset(ctrl, "mButtonFrameFire", GameGetFrameNum()) end
    end
end

-- ── Respawn: restore HP and teleport to recorded spawn ───────────────────
local function respawn_player(player)
    local dmg = EntityGetFirstComponent(player, "DamageModelComponent")
    if dmg then
        local max_hp = cget(dmg, "max_hp") or 4
        cset(dmg, "hp", max_hp)
    end
    if spawn_x then
        local ok, e = pcall(EntitySetTransform, player, spawn_x, spawn_y)
        if not ok then warn("EntitySetTransform failed: " .. tostring(e)) end
    end
    local cdata = EntityGetFirstComponent(player, "CharacterDataComponent")
    if cdata then cset(cdata, "mVelocity", 0, 0) end

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

-- ── 5 downward liquid sensors ─────────────────────────────────────────────
-- For each angle: RaytraceSurfacesAndLiquiform hits liquid surface,
-- RaytracePlatforms passes through liquid to solid floor.
-- Signal = (d_solid - d_liquid) / ray_len; 0 = dry, ~1 = pool right there.
local LIQUID_ANGLES = {
    math.pi * 1/4,  -- down-right
    math.pi * 1/3,
    math.pi / 2,    -- straight down
    math.pi * 2/3,
    math.pi * 3/4,  -- down-left
}
local LIQUID_LEN = 80

local function build_liquid_sensors(x, y)
    local out = {}
    for _, angle in ipairs(LIQUID_ANGLES) do
        local tx = x + math.cos(angle) * LIQUID_LEN
        local ty = y + math.sin(angle) * LIQUID_LEN

        local ok1, hit1, hx1, hy1 = pcall(RaytraceSurfacesAndLiquiform, x, y, tx, ty)
        local ok2, hit2, hx2, hy2 = pcall(RaytracePlatforms,            x, y, tx, ty)

        local d1 = (ok1 and hit1) and math.sqrt((hx1-x)^2+(hy1-y)^2) or LIQUID_LEN
        local d2 = (ok2 and hit2) and math.sqrt((hx2-x)^2+(hy2-y)^2) or LIQUID_LEN

        table.insert(out, math.max(0.0, (d2 - d1) / LIQUID_LEN))
    end
    return out
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

    local hp = 1.0
    local dmg = EntityGetFirstComponent(player, "DamageModelComponent")
    if dmg then
        local raw_hp  = cget(dmg, "hp")    or 0
        local raw_max = cget(dmg, "max_hp") or 1
        if raw_max > 0 then hp = raw_hp / raw_max end
    end

    local vx, vy, on_ground = 0.0, 0.0, false
    local cdata = EntityGetFirstComponent(player, "CharacterDataComponent")
    if cdata then
        vx, vy   = cget(cdata, "mVelocity")
        on_ground = cget(cdata, "is_on_ground")
        vx = vx or 0; vy = vy or 0; on_ground = on_ground or false
    end

    local rays           = build_rays(x, y)
    local liquid_sensors = build_liquid_sensors(x, y)

    -- Draw radar ───────────────────────────────────────────────────────────
    GuiIdPushString(gui, "rl_radar")
    draw_radar(gui, 10, 24, rays)
    GuiIdPop(gui)

    episode_steps = episode_steps + 1

    -- Periodic diagnostic log ──────────────────────────────────────────────
    if (frame % 300) == 0 then
        info(string.format(
            "DIAG ep=%d step=%d pos=(%.0f,%.0f) vel=(%.1f,%.1f) hp=%.2f gnd=%s act=%s fps=%.0f",
            episode_num, episode_steps, x, y, vx, vy, hp,
            tostring(on_ground), ACTION_NAMES[last_action] or "?", fps))
    end

    -- Death detection ──────────────────────────────────────────────────────
    if hp <= DEATH_HP then
        info(string.format("Ep %d ended  steps=%d  pos=(%.0f,%.0f)  hp=%.3f",
            episode_num, episode_steps, x, y, hp))
        local dead_state = {
            x=x, y=y, hp=0.0, vx=0.0, vy=0.0,
            rays=rays, dead=true, on_ground=false
        }
        local ok4, encoded = pcall(json.encode, dead_state)
        if ok4 then
            pcall(socket.send, socket, encoded)
        end
        respawn_player(player)
        return
    end

    -- Send live state ──────────────────────────────────────────────────────
    local state = {x=x, y=y, hp=hp, vx=vx, vy=vy, rays=rays, liquid_sensors=liquid_sensors, dead=false, on_ground=on_ground, frame=frame}
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
