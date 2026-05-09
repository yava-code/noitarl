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

-- ── Connection ────────────────────────────────────────────────────────────
local socket                    = nil
local last_connection_attempt   = 0
local connection_retry_interval = 180   -- frames (~3 sec at 60fps)
local gui                       = nil

-- ── Movement ──────────────────────────────────────────────────────────────
local MOVE_SPEED      = 60
local JUMP_SPEED      = -150
local pending_action  = 0
local last_action     = 0
local ACTION_NAMES    = {[0]="IDLE",[1]="LEFT",[2]="RIGHT",[3]="JUMP",[4]="FIRE"}

-- ── Episode state ─────────────────────────────────────────────────────────
local spawn_x, spawn_y  = nil, nil
local episode_num       = 0
local episode_steps     = 0
local DEATH_HP          = 0.02   -- respawn when hp fraction falls below this

-- ── Apply action via direct velocity write ────────────────────────────────
-- Writing to CharacterDataComponent.mVelocity bypasses the input pipeline;
-- ControlsComponent button fields are overwritten by the engine's keyboard
-- reader each frame and don't survive to the physics tick.
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

    -- Fire goes through ControlsComponent (not velocity-based)
    local ctrl = EntityGetFirstComponent(player, "ControlsComponent")
    if ctrl then
        local fire = (action == 4)
        ComponentSetValue2(ctrl, "mButtonDownFire", fire)
        if fire then ComponentSetValue2(ctrl, "mButtonFrameFire", GameGetFrameNum()) end
    end
end

-- ── Respawn: restore HP and teleport to spawn position ───────────────────
local function respawn_player(player)
    local dmg = EntityGetFirstComponent(player, "DamageModelComponent")
    if dmg then
        ComponentSetValue2(dmg, "hp", ComponentGetValue2(dmg, "max_hp"))
    end
    if spawn_x then
        EntitySetTransform(player, spawn_x, spawn_y)
    end
    local cdata = EntityGetFirstComponent(player, "CharacterDataComponent")
    if cdata then ComponentSetValue2(cdata, "mVelocity", 0, 0) end

    episode_num    = episode_num + 1
    episode_steps  = 0
    pending_action = 0
    log(string.format("Ep %d started — spawn (%.0f, %.0f)", episode_num, spawn_x or 0, spawn_y or 0))
end

-- ── Ray visualisation (dev build only) ───────────────────────────────────
-- Each ray: green endpoint = open space, red = wall nearby.
-- Two dots per ray (50% and 100%) create a dotted-line effect.
local function draw_rays(ox, oy, rays)
    if not DEBUG_MARK then return end
    for i = 0, 15 do
        local angle = i * (math.pi / 8)
        local dist  = rays[i + 1]
        local len   = dist * 150
        local r, g  = 1.0 - dist, dist
        DEBUG_MARK(ox + math.cos(angle)*len*0.5, oy + math.sin(angle)*len*0.5, "", r*0.5, g*0.5, 0)
        DEBUG_MARK(ox + math.cos(angle)*len,     oy + math.sin(angle)*len,     "", r,     g,     0)
    end
end

-- ── Pre-update: apply buffered action BEFORE physics simulation ───────────
function OnWorldPreUpdate()
    if not RaytracePlatforms then return end
    local player = EntityGetWithTag("player_unit")[1]
    if player then apply_action(player, pending_action) end
end

-- ── Post-update: gather state, draw HUD, communicate with Python ──────────
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
        GuiText(gui, 10, 10, string.format("RL AGENT: DISCONNECTED  retry in %d", retry))
    else
        GuiText(gui, 10, 10, string.format(
            "RL AGENT  Ep:%-3d  Step:%-5d  Act: %s",
            episode_num, episode_steps, ACTION_NAMES[pending_action] or "?"))
        GuiText(gui, 10, 20, string.format(
            "socket: %s   spawn: (%.0f, %.0f)",
            socket:status(), spawn_x or 0, spawn_y or 0))
    end
    GuiIdPop(gui)

    -- Connection management ────────────────────────────────────────────────
    if not socket then
        if frame - last_connection_attempt > connection_retry_interval then
            log("Connecting to ws://localhost:5001")
            last_connection_attempt = frame
            socket = pollnet.open_ws("ws://localhost:5001")
        end
        return
    end

    local _, msg  = socket:poll()
    local st      = socket:status()

    if st == "error" or st == "closed" then
        log("Socket " .. st .. " — will reconnect")
        pending_action = 0
        socket = nil
        return
    end

    -- Buffer incoming action for next frame's pre-update ──────────────────
    if msg and type(msg) == "string" then
        local ok, act = pcall(json.decode, msg)
        if ok and type(act) == "number" then pending_action = act end
    end

    if st ~= "open" or not player then return end

    -- Gather player state ──────────────────────────────────────────────────
    local x, y = EntityGetTransform(player)

    if not spawn_x then                  -- record once on first live frame
        spawn_x, spawn_y = x, y
        episode_num = 1
        log(string.format("Spawn recorded (%.0f, %.0f)", x, y))
    end

    local hp = 1.0
    local dmg = EntityGetFirstComponent(player, "DamageModelComponent")
    if dmg then
        local raw_max = ComponentGetValue2(dmg, "max_hp")
        if raw_max and raw_max > 0 then
            hp = ComponentGetValue2(dmg, "hp") / raw_max
        end
    end

    local vx, vy, on_ground = 0.0, 0.0, false
    local cdata = EntityGetFirstComponent(player, "CharacterDataComponent")
    if cdata then
        vx, vy   = ComponentGetValue2(cdata, "mVelocity")
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
        rays[i + 1] = hit and math.sqrt((hx-x)^2 + (hy-y)^2) / 150 or 1.0
    end

    draw_rays(x, y, rays)
    episode_steps = episode_steps + 1

    -- Death detection ──────────────────────────────────────────────────────
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

    -- Send live state ──────────────────────────────────────────────────────
    socket:send(json.encode({
        x=x, y=y, hp=hp, vx=vx, vy=vy,
        rays=rays, dead=false, on_ground=on_ground
    }))
end
