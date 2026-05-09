-- mods/rl_agent/init.lua

local function log(msg)
    local timestamp = os.date("%Y-%m-%d %H:%M:%S")
    local formatted_msg = string.format("[%s] [RL_AGENT] %s", timestamp, tostring(msg))
    print(formatted_msg)
    local f = io.open("mods/rl_agent/logger.txt", "a")
    if f then
        f:write(formatted_msg .. "\n")
        f:close()
    end
end

log("Mod init started")

local function get_mod_path()
    return "mods/rl_agent/"
end

local json = dofile(get_mod_path() .. "lib/json.lua")
local pollnet = dofile(get_mod_path() .. "lib/pollnet.lua")

local socket = nil
local last_connection_attempt = 0
local connection_retry_interval = 180
local gui = nil

local function apply_action(player, action)
    local cc = EntityGetFirstComponent(player, "CharacterPlatformingComponent")
    if cc then
        ComponentSetValue2(cc, "mMoveLeftDown",  action == 1)
        ComponentSetValue2(cc, "mMoveRightDown", action == 2)
        -- jump: give the character jump frames
        if action == 3 then
            ComponentSetValue2(cc, "mJumpFramesLeft", 10)
        end
    end

    local ctrl = EntityGetFirstComponent(player, "ControlsComponent")
    if ctrl then
        ComponentSetValue2(ctrl, "mButtonFireDownLeft", action == 4)
    end
end

function OnWorldPostUpdate()
    if not RaycastPlatforms then return end

    local frame = GameGetFrameNum()
    local player = EntityGetWithTag("player_unit")[1]

    -- reuse one GUI object per frame
    if not gui then gui = GuiCreate() end
    local status_text = "RL AGENT: "
    if not socket then
        local retry_in = math.max(0, connection_retry_interval - (frame - last_connection_attempt))
        status_text = status_text .. "DISCONNECTED (retry in " .. retry_in .. ")"
    else
        status_text = status_text .. "STATE: " .. tostring(socket:status())
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
            local hp = 1.0
            local dmg = EntityGetFirstComponent(player, "DamageModelComponent")
            if dmg then
                hp = ComponentGetValue2(dmg, "hp") / ComponentGetValue2(dmg, "max_hp")
            end

            local rays = {}
            for i = 0, 15 do
                local angle = i * (math.pi / 8)
                local hit, hx, hy = RaycastPlatforms(
                    x, y,
                    x + math.cos(angle) * 150,
                    y + math.sin(angle) * 150
                )
                local dist = hit and (math.sqrt((hx - x)^2 + (hy - y)^2) / 150) or 1.0
                table.insert(rays, dist)
            end

            socket:send(json.encode({ x = x, y = y, hp = hp, vx = 0, vy = 0, rays = rays }))
        end
    elseif state == "error" or state == "closed" then
        log("Socket " .. state .. ", will reconnect")
        socket = nil
        return
    end

    if msg and type(msg) == "string" then
        local ok, action = pcall(json.decode, msg)
        if ok and type(action) == "number" and player then
            apply_action(player, action)
        end
    end
end
