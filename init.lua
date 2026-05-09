-- mods/noitarl/init.lua

local function log(msg)
    local ts  = os.date("%H:%M:%S")
    local out = string.format("[%s] [NOITARL] %s", ts, tostring(msg))
    print(out)
    local f = io.open("mods/noitarl/logger.txt", "a")
    if f then f:write(out .. "\n"); f:close() end
end

log("Mod init started")

local json    = dofile("mods/noitarl/lib/json.lua")
local pollnet = dofile("mods/noitarl/lib/pollnet.lua")

-- ── Config (edit port.txt to change port for multi-instance setups) ───────
local function read_port()
    local f = io.open("mods/noitarl/port.txt", "r")
    if f then local p = tonumber(f:read("*l")); f:close(); return p or 5001 end
    return 5001
end
local WS_PORT = read_port()
local WS_URL  = "ws://localhost:" .. WS_PORT

-- ── Connection ────────────────────────────────────────────────────────────
local socket                    = nil
local last_connection_attempt   = 0
local connection_retry_interval = 180
local gui                       = nil

-- ── Movement ──────────────────────────────────────────────────────────────
local MOVE_SPEED     = 60
local JUMP_SPEED     = -150
local pending_action = 0
local last_action    = 0
local ACTION_NAMES   = {[0]="IDLE",[1]="LEFT",[2]="RIGHT",[3]="JUMP",[4]="FIRE"}

-- ── Episode state ─────────────────────────────────────────────────────────
local spawn_x, spawn_y = nil, nil
local episode_num      = 0
local episode_steps    = 0
local DEATH_HP         = 0.02

-- ── Action via direct velocity write ─────────────────────────────────────
-- ControlsComponent.mButtonDown* is overwritten by the engine's keyboard
-- reader every frame before physics runs; mVelocity is what physics uses.
local function apply_action(player, action)
    if not player or player == 0 then return end
    last_action = action

    local cdata = EntityGetFirstComponent(player, "CharacterDataComponent")
    if cdata then
        local vx, vy    = ComponentGetValue2(cdata, "mVelocity")
        local on_ground = ComponentGetValue2(cdata, "is_on_ground")
        if     action == 1 then vx = -MOVE_SPEED
        elseif action == 2 then vx =  MOVE_SPEED
        elseif action == 3 and on_ground then vy = JUMP_SPEED
        end
        ComponentSetValue2(cdata, "mVelocity", vx, vy)
    end

    local ctrl = EntityGetFirstComponent(player, "ControlsComponent")
    if ctrl then
        local fire = (action == 4)
        ComponentSetValue2(ctrl, "mButtonDownFire", fire)
        if fire then ComponentSetValue2(ctrl, "mButtonFrameFire", GameGetFrameNum()) end
    end
end

-- ── Respawn ───────────────────────────────────────────────────────────────
local function respawn_player(player)
    local dmg = EntityGetFirstComponent(player, "DamageModelComponent")
    if dmg then ComponentSetValue2(dmg, "hp", ComponentGetValue2(dmg, "max_hp")) end
    if spawn_x then EntitySetTransform(player, spawn_x, spawn_y) end
    local cdata = EntityGetFirstComponent(player, "CharacterDataComponent")
    if cdata then ComponentSetValue2(cdata, "mVelocity", 0, 0) end
    episode_num    = episode_num + 1
    episode_steps  = 0
    pending_action = 0
    log(string.format("Ep %d started — spawn (%.0f, %.0f)", episode_num, spawn_x or 0, spawn_y or 0))
end

-- ── Mini radar HUD (no world-space rendering → zero FPS cost) ─────────────
-- 3×3 grid showing 8 directions.  Ray index→direction (angle = i*π/8):
--   E=1  SE=3  S=5  SW=7  W=9  NW=11  N=13  NE=15  (1-indexed in rays table)
-- Characters: # close  + medium  . far  space=open
local function dist_char(d)
    if d < 0.25 then return "#"
    elseif d < 0.5  then return "+"
    elseif d < 0.75 then return "."
    else return " " end
end

local function draw_radar(gui_obj, x, y, rays)
    -- grid[row][col] = ray table index, or -1 for player marker
    local grid = {
        { 11, 13, 15 },   -- NW  N  NE
        {  9, -1,  1 },   -- W   @   E
        {  7,  5,  3 },   -- SW  S  SE
    }
    local cell = 8   -- pixels per cell
    for row = 1, 3 do
        for col = 1, 3 do
            local idx = grid[row][col]
            local tx  = x + (col - 1) * cell
            local ty  = y + (row - 1) * cell
            if idx == -1 then
                GuiColorSetForNextWidget(gui_obj, 0, 1, 1, 1)
                GuiText(gui_obj, tx, ty, "@")
            else
                local d = rays[idx]
                -- red = wall close, green = open space
                GuiColorSetForNextWidget(gui_obj, 1 - d, d, 0, 1)
                GuiText(gui_obj, tx, ty, dist_char(d))
            end
        end
    end
    -- reset colour
    GuiColorSetForNextWidget(gui_obj, 1, 1, 1, 1)
    GuiText(gui_obj, x, y + 3 * cell, string.format("port:%d", WS_PORT))
end

-- ── Pre-update: apply buffered action BEFORE physics ─────────────────────
function OnWorldPreUpdate()
    if not RaytracePlatforms then return end
    local player = EntityGetWithTag("player_unit")[1]
    if player then apply_action(player, pending_action) end
end

-- ── Post-update: state → Python, action ← Python ─────────────────────────
function OnWorldPostUpdate()
    if not RaytracePlatforms then return end

    local frame  = GameGetFrameNum()
    local player = EntityGetWithTag("player_unit")[1]

    -- HUD ──────────────────────────────────────────────────────────────────
    if not gui then gui = GuiCreate() end
    GuiStartFrame(gui)
    GuiIdPushString(gui, "rl_hud")

    if not socket then
        local retry = math.max(0, connection_retry_interval - (frame - last_connection_attempt))
        GuiColorSetForNextWidget(gui, 1, 0.4, 0.2, 1)
        GuiText(gui, 10, 10, string.format("RL AGENT  DISCONNECTED  retry:%d", retry))
        GuiColorSetForNextWidget(gui, 1, 1, 1, 1)
    else
        local act = ACTION_NAMES[pending_action] or "?"
        GuiColorSetForNextWidget(gui, 0.4, 1, 0.4, 1)
        GuiText(gui, 10, 10, string.format("RL AGENT  Ep:%-3d  Step:%-5d  %s",
            episode_num, episode_steps, act))
        GuiColorSetForNextWidget(gui, 1, 1, 1, 1)
        GuiText(gui, 10, 20, string.format("socket: %s", socket:status()))
    end

    GuiIdPop(gui)

    -- Connection ───────────────────────────────────────────────────────────
    if not socket then
        if frame - last_connection_attempt > connection_retry_interval then
            log("Connecting → " .. WS_URL)
            last_connection_attempt = frame
            socket = pollnet.open_ws(WS_URL)
        end
        return
    end

    local _, msg = socket:poll()
    local st     = socket:status()

    if st == "error" or st == "closed" then
        log("Socket " .. st .. " — reconnecting")
        pending_action = 0; socket = nil; return
    end

    if msg and type(msg) == "string" then
        local ok, act = pcall(json.decode, msg)
        if ok and type(act) == "number" then pending_action = act end
    end

    if st ~= "open" or not player then return end

    -- Player state ─────────────────────────────────────────────────────────
    local x, y = EntityGetTransform(player)

    if not spawn_x then
        spawn_x, spawn_y = x, y
        episode_num = 1
        log(string.format("Spawn (%.0f, %.0f)", x, y))
    end

    local hp = 1.0
    local dmg = EntityGetFirstComponent(player, "DamageModelComponent")
    if dmg then
        local mh = ComponentGetValue2(dmg, "max_hp")
        if mh and mh > 0 then hp = ComponentGetValue2(dmg, "hp") / mh end
    end

    local vx, vy, on_ground = 0.0, 0.0, false
    local cdata = EntityGetFirstComponent(player, "CharacterDataComponent")
    if cdata then
        vx, vy    = ComponentGetValue2(cdata, "mVelocity")
        on_ground = ComponentGetValue2(cdata, "is_on_ground")
    end

    -- 16 raytrace sensors ──────────────────────────────────────────────────
    local rays = {}
    for i = 0, 15 do
        local angle = i * (math.pi / 8)
        local hit, hx, hy = RaytracePlatforms(
            x, y,
            x + math.cos(angle) * 150,
            y + math.sin(angle) * 150
        )
        rays[i + 1] = hit and math.sqrt((hx-x)^2+(hy-y)^2)/150 or 1.0
    end

    -- Draw lightweight radar (no world-space markers) ──────────────────────
    GuiStartFrame(gui)
    GuiIdPushString(gui, "rl_radar")
    draw_radar(gui, 10, 34, rays)
    GuiIdPop(gui)

    episode_steps = episode_steps + 1

    -- Death / reset ────────────────────────────────────────────────────────
    if hp <= DEATH_HP then
        log(string.format("Ep %d ended  steps=%d  pos=(%.0f,%.0f)",
            episode_num, episode_steps, x, y))
        socket:send(json.encode({
            x=x, y=y, hp=0.0, vx=0.0, vy=0.0,
            rays=rays, dead=true, on_ground=false
        }))
        respawn_player(player)
        return
    end

    socket:send(json.encode({
        x=x, y=y, hp=hp, vx=vx, vy=vy,
        rays=rays, dead=false, on_ground=on_ground
    }))
end
