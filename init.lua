-- mods/noitarl/init.lua

local function log(msg)
    local timestamp = os.date("%Y-%m-%d %H:%M:%S")
    local formatted_msg = string.format("[%s] [NOITARL] %s", timestamp, tostring(msg))
    print(formatted_msg)
    local f = io.open("mods/noitarl/logger.txt", "a")
    if f then
        f:write(formatted_msg .. "\n")
        f:close()
    end
end

log("Mod init started")

local function get_mod_path()
    return "mods/noitarl/"
end

local json    = dofile(get_mod_path() .. "lib/json.lua")
local pollnet = dofile(get_mod_path() .. "lib/pollnet.lua")

local socket                   = nil
local last_connection_attempt  = 0
local connection_retry_interval = 180
local gui                      = nil
local pending_action           = 0  -- last action from Python, applied in pre-update

local MOVE_SPEED          = 50    -- horizontal velocity (Noita units)
local JUMP_SPEED          = -150  -- negative = up
local DIAG_LOG_EVERY      = 60    -- frames between diag dumps
local last_applied_action = 0

local function apply_action(player, action)
    if not player or player == 0 then return end
    last_applied_action = action

    -- Direct velocity write — bypasses input pipeline that overrides ControlsComponent every frame
    local cdata = EntityGetFirstComponent(player, "CharacterDataComponent")
    if cdata then
        local vx, vy    = ComponentGetValue2(cdata, "mVelocity")
        local on_ground = ComponentGetValue2(cdata, "is_on_ground")

        if action == 1 then            -- LEFT
            vx = -MOVE_SPEED
        elseif action == 2 then        -- RIGHT
            vx =  MOVE_SPEED
        elseif action == 3 and on_ground then  -- JUMP only when grounded
            vy = JUMP_SPEED
        end
        ComponentSetValue2(cdata, "mVelocity", vx, vy)
    end

    -- Fire still goes through ControlsComponent (not velocity-based)
    local ctrl = EntityGetFirstComponent(player, "ControlsComponent")
    if ctrl then
        local fire = (action == 4)
        ComponentSetValue2(ctrl, "mButtonDownFire", fire)
        if fire then
            ComponentSetValue2(ctrl, "mButtonFrameFire", GameGetFrameNum())
        end
    end
end

-- Apply buffered action BEFORE the simulation runs this frame
function OnWorldPreUpdate()
    if not RaytracePlatforms then return end
    local player = EntityGetWithTag("player_unit")[1]
    if player then
        apply_action(player, pending_action)

        local frame = GameGetFrameNum()
        if (frame % DIAG_LOG_EVERY) == 0 then
            local px, py = EntityGetTransform(player)
            local cdata  = EntityGetFirstComponent(player, "CharacterDataComponent")
            local vx, vy, grounded = 0, 0, false
            if cdata then
                vx, vy   = ComponentGetValue2(cdata, "mVelocity")
                grounded = ComponentGetValue2(cdata, "is_on_ground")
            end
            log(string.format(
                "DIAG f=%d player=%d pos=(%.1f,%.1f) vel=(%.2f,%.2f) on_ground=%s action=%d cdata=%s",
                frame, player, px, py, vx, vy, tostring(grounded),
                last_applied_action, tostring(cdata ~= nil)
            ))
        end
    end
end

-- After simulation: send state to Python, receive next action into buffer
function OnWorldPostUpdate()
    if not RaytracePlatforms then return end

    local frame  = GameGetFrameNum()
    local player = EntityGetWithTag("player_unit")[1]

    if not gui then gui = GuiCreate() end
    local status_text = "RL AGENT: "
    if not socket then
        local retry_in = math.max(0, connection_retry_interval - (frame - last_connection_attempt))
        status_text = status_text .. "DISCONNECTED (retry in " .. retry_in .. ")"
    else
        status_text = status_text .. "STATE: " .. tostring(socket:status())
                   .. " | ACT: " .. tostring(pending_action)
    end
    GuiStartFrame(gui)
    GuiIdPushString(gui, "rl_debug")
    GuiText(gui, 10, 10, status_text)
    GuiIdPop(gui)

    if not socket then
        if frame - last_connection_attempt > connection_retry_interval then
            log("Attempting connect to ws://localhost:5001")
            last_connection_attempt = frame
            socket = pollnet.open_ws("ws://localhost:5001")
        end
        return
    end

    local happy, msg = socket:poll()
    local state = socket:status()

    if state == "open" then
        if player then
            local x, y = EntityGetTransform(player)
            local hp   = 1.0
            local dmg  = EntityGetFirstComponent(player, "DamageModelComponent")
            if dmg then
                hp = ComponentGetValue2(dmg, "hp") / ComponentGetValue2(dmg, "max_hp")
            end

            local vx, vy = 0, 0
            local cdata  = EntityGetFirstComponent(player, "CharacterDataComponent")
            if cdata then
                vx, vy = ComponentGetValue2(cdata, "mVelocity")
            end

            local rays = {}
            for i = 0, 15 do
                local angle = i * (math.pi / 8)
                local hit, hx, hy = RaytracePlatforms(
                    x, y,
                    x + math.cos(angle) * 150,
                    y + math.sin(angle) * 150
                )
                local dist = hit and (math.sqrt((hx - x)^2 + (hy - y)^2) / 150) or 1.0
                table.insert(rays, dist)
            end

            socket:send(json.encode({ x = x, y = y, hp = hp, vx = vx, vy = vy, rays = rays }))
        end
    elseif state == "error" or state == "closed" then
        log("Socket " .. state .. ", will reconnect")
        pending_action = 0
        socket = nil
        return
    end

    -- Buffer the received action for next frame's pre-update
    if msg and type(msg) == "string" then
        local ok, action = pcall(json.decode, msg)
        if ok and type(action) == "number" then
            pending_action = action
        end
    end
end
