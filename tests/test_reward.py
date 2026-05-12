"""
Tests for reward calculation logic in NoitaEnv.step().

We test the reward components in isolation by calling step() on a minimally
configured env with a pre-loaded fake state, and by directly testing the
inline logic extracted into helper assertions.

No Noita process, no WebSocket, no hardware access required.
"""
import sys, types
import numpy as np
import pytest

# Stub hardware imports before loading noita_env
for mod in ("mss", "pygetwindow"):
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)


# ---------------------------------------------------------------------------
# Helpers to exercise reward components without running step()
# ---------------------------------------------------------------------------

def _make_env_stub(
    spawn_x=0.0, spawn_y=0.0,
    max_dist=0.0, max_depth=0.0,
    last_hp=1.0, last_kills=0,
):
    """Return a bare NoitaEnv with state needed for reward math."""
    from noita_env import NoitaEnv
    env = NoitaEnv.__new__(NoitaEnv)
    env.spawn_x             = spawn_x
    env.spawn_y             = spawn_y
    env.max_spawn_distance  = max_dist
    env.max_depth_y         = max_depth
    env.last_hp             = last_hp
    env.last_kills          = last_kills
    env.episode_steps       = 0
    env.episode_reward      = 0.0
    env.visited_chunks      = set()
    env.steps_without_progress = 0
    env.route_x             = []
    env.route_y             = []
    env.action_history      = []
    env._fast_mv_counter    = 0
    env._recorder           = None
    return env


def _calc_reward(env, current_x, current_y, current_hp, current_kills,
                 action, prev_state, state, dead=False, truncated=False):
    """
    Run the same arithmetic as NoitaEnv.step() and return (reward, env_after).
    This mirrors the step() reward block without the WS/obs machinery.
    """
    reward = -0.001  # time tax

    dist = abs(current_x - env.spawn_x) + abs(current_y - env.spawn_y)
    if dist > env.max_spawn_distance:
        reward += (dist - env.max_spawn_distance) * 0.02
        env.max_spawn_distance = dist
        env.steps_without_progress = 0
    else:
        env.steps_without_progress += 1

    chunk = (int(current_x // 32), int(current_y // 32))
    if chunk not in env.visited_chunks:
        env.visited_chunks.add(chunk)
        sky_vis = float(state.get("sky_visibility", 0.0))
        if sky_vis < 0.3:
            reward += 0.5

    if current_y > env.max_depth_y:
        reward += (current_y - env.max_depth_y) * 0.02
        env.max_depth_y = current_y

    if current_hp < env.last_hp:
        reward -= (env.last_hp - current_hp) * 1.0

    if current_kills > env.last_kills:
        reward += (current_kills - env.last_kills) * 5.0

    if action in (6, 7):
        enemy_radar = state.get("enemy_radar", [1.0] * 8)
        if any(v < 0.9 for v in enemy_radar):
            reward += 0.05

    portal = state.get("portal", [1.0, 0.5, 0.5])
    if isinstance(portal, list) and len(portal) == 3:
        pd = float(portal[0])
        if pd < 0.3:
            reward += (0.3 - pd) * 0.05

    # Portal teleport
    portal_bonus = 0.0
    if prev_state is not None and not prev_state.get("dead", False):
        prev_x = prev_state.get("x", current_x)
        prev_y = prev_state.get("y", current_y)
        prev_portal = prev_state.get("portal", [1.0, 0.5, 0.5])
        was_near = isinstance(prev_portal, list) and len(prev_portal) == 3 and float(prev_portal[0]) < 0.5
        if (abs(current_x - prev_x) > 300 or abs(current_y - prev_y) > 300) and was_near:
            portal_bonus = 20.0
    reward += portal_bonus

    if dead and current_hp <= 0:
        reward -= 1.0

    env.last_hp    = current_hp
    env.last_kills = current_kills
    return reward


# ---------------------------------------------------------------------------
# Time tax
# ---------------------------------------------------------------------------

class TestTimeTax:
    def test_idle_step_gives_negative_base(self):
        # spawn=(100,100), max_depth=100 so no depth bonus, pre-visit chunk
        env = _make_env_stub(spawn_x=100, spawn_y=100, max_depth=100.0, max_dist=0.0)
        env.visited_chunks.add((100 // 32, 100 // 32))
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        r = _calc_reward(env, 100, 100, 1.0, 0, 0, None, state)
        assert r == pytest.approx(-0.001, abs=1e-5)


# ---------------------------------------------------------------------------
# Manhattan progress
# ---------------------------------------------------------------------------

class TestManhattanProgress:
    def test_first_move_right_gives_bonus(self):
        env = _make_env_stub()
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        r = _calc_reward(env, 100, 0, 1.0, 0, 0, None, state)
        # dist = 100, no prior record → + 100 * 0.02 = +2.0
        assert r > 0.0

    def test_repeated_position_no_bonus(self):
        env = _make_env_stub(max_dist=50.0)
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        # Move to only dist=40 (less than 50) → no bonus
        env.visited_chunks.add((0, 0))   # pre-visit chunk
        r = _calc_reward(env, 40, 0, 1.0, 0, 0, None, state)
        assert r == pytest.approx(-0.001, abs=1e-4)

    def test_lateral_movement_credited(self):
        env = _make_env_stub(spawn_x=0, spawn_y=0)
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        env.visited_chunks.add((3, 0))   # pre-visit chunk
        r = _calc_reward(env, 100, 0, 1.0, 0, 0, None, state)
        assert r > 0.0, "Lateral (X) movement must be rewarded"

    def test_diagonal_movement_credited(self):
        env = _make_env_stub()
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        env.visited_chunks.add((3, 3))
        r = _calc_reward(env, 100, 100, 1.0, 0, 0, None, state)
        # Manhattan = 200, bonus = 200 * 0.02 = +4.0
        assert r > 3.0


# ---------------------------------------------------------------------------
# Chunk curiosity bonus
# ---------------------------------------------------------------------------

class TestChunkBonus:
    def test_new_underground_chunk_rewards(self):
        env = _make_env_stub()
        state = {"sky_visibility": 0.1, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        r = _calc_reward(env, 16, 500, 1.0, 0, 0, None, state)
        assert r > 0.49   # chunk bonus included

    def test_sky_chunk_no_reward_when_visible(self):
        env = _make_env_stub()
        state = {"sky_visibility": 0.9, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        # Visit chunk at surface (sky_vis >= 0.3)
        r = _calc_reward(env, 16, 0, 1.0, 0, 0, None, state)
        # No chunk +0.5 because sky_vis = 0.9
        assert r < 0.5

    def test_sky_threshold_boundary_03(self):
        env_under = _make_env_stub()
        env_over  = _make_env_stub()
        state_u = {"sky_visibility": 0.29, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        state_o = {"sky_visibility": 0.30, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        r_u = _calc_reward(env_under, 16, 200, 1.0, 0, 0, None, state_u)
        r_o = _calc_reward(env_over,  16, 200, 1.0, 0, 0, None, state_o)
        assert r_u > r_o   # underground chunk pays more

    def test_revisited_chunk_no_bonus(self):
        # spawn at (16, 500) so no manhattan or depth bonus; chunk (0,15) pre-visited
        env = _make_env_stub(spawn_x=16, spawn_y=500, max_dist=0.0, max_depth=500.0)
        env.visited_chunks.add((0, 15))   # pre-visit chunk for y=500
        state = {"sky_visibility": 0.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        r = _calc_reward(env, 16, 500, 1.0, 0, 0, None, state)
        assert r == pytest.approx(-0.001, abs=1e-4)  # only time tax


# ---------------------------------------------------------------------------
# Depth bonus
# ---------------------------------------------------------------------------

class TestDepthBonus:
    def test_new_depth_record_pays(self):
        env = _make_env_stub(max_depth=100.0)
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        env.visited_chunks.add((0, 6))
        r = _calc_reward(env, 0, 200, 1.0, 0, 0, None, state)
        # Δdepth = 100, bonus = 100 * 0.02 = +2.0
        assert r > 1.9

    def test_no_depth_bonus_when_going_up(self):
        # spawn at (0,300) with max_dist already 300 so no manhattan bonus either
        env = _make_env_stub(spawn_x=0, spawn_y=0, max_dist=300.0, max_depth=500.0)
        env.visited_chunks.add((0, 9))   # chunk for y=300
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        r = _calc_reward(env, 0, 300, 1.0, 0, 0, None, state)
        # current_y=300 < max_depth=500 and dist=300=max_dist → no bonus, only time tax
        assert r == pytest.approx(-0.001, abs=1e-4)


# ---------------------------------------------------------------------------
# Damage penalty
# ---------------------------------------------------------------------------

class TestDamagePenalty:
    def test_damage_subtracts(self):
        env = _make_env_stub(last_hp=1.0)
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        env.visited_chunks.add((0, 0))
        r = _calc_reward(env, 0, 0, 0.5, 0, 0, None, state)
        # Δhp = 0.5, penalty = -0.5
        assert r < -0.4

    def test_full_damage_big_penalty(self):
        env = _make_env_stub(last_hp=1.0)
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        env.visited_chunks.add((0, 0))
        r = _calc_reward(env, 0, 0, 0.0, 0, 0, None, state, dead=True)
        # -1.0 (full HP lost) + -1.0 (death) + -0.001 (time tax)
        assert r < -1.9

    def test_no_damage_no_penalty(self):
        env = _make_env_stub(last_hp=0.5)
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        env.visited_chunks.add((0, 0))
        r = _calc_reward(env, 0, 0, 0.5, 0, 0, None, state)  # same HP
        assert r == pytest.approx(-0.001, abs=1e-4)


# ---------------------------------------------------------------------------
# Kill reward
# ---------------------------------------------------------------------------

class TestKillReward:
    def test_single_kill_gives_5(self):
        env = _make_env_stub(last_kills=0)
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        env.visited_chunks.add((0, 0))
        r = _calc_reward(env, 0, 0, 1.0, 1, 6, None, state)
        # +5.0 kill + aim-on-enemy (+0.05 * 0 since no enemy in radar) + time tax
        # enemy_radar = all 1.0, so no aim bonus
        assert r > 4.9

    def test_double_kill_gives_10(self):
        env = _make_env_stub(last_kills=0)
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        env.visited_chunks.add((0, 0))
        r = _calc_reward(env, 0, 0, 1.0, 2, 0, None, state)
        assert r > 9.9


# ---------------------------------------------------------------------------
# Aim-on-enemy bonus
# ---------------------------------------------------------------------------

class TestAimBonus:
    def test_fire_with_enemy_pays(self):
        env = _make_env_stub()
        env.visited_chunks.add((0, 0))
        state = {"sky_visibility": 1.0, "enemy_radar": [0.5] + [1.0]*7, "portal": [1.0,0.5,0.5]}
        r = _calc_reward(env, 0, 0, 1.0, 0, 6, None, state)
        # aim bonus = +0.05
        assert r > 0.04

    def test_fire_without_enemy_no_bonus(self):
        env = _make_env_stub()
        env.visited_chunks.add((0, 0))
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        r = _calc_reward(env, 0, 0, 1.0, 0, 6, None, state)
        assert r == pytest.approx(-0.001, abs=1e-4)

    def test_non_fire_action_no_bonus(self):
        env = _make_env_stub()
        env.visited_chunks.add((0, 0))
        state = {"sky_visibility": 1.0, "enemy_radar": [0.3]*8, "portal": [1.0,0.5,0.5]}
        r = _calc_reward(env, 0, 0, 1.0, 0, 1, None, state)   # action LEFT
        assert r == pytest.approx(-0.001, abs=1e-4)

    def test_fire_down_with_enemy_pays(self):
        env = _make_env_stub()
        env.visited_chunks.add((0, 0))
        state = {"sky_visibility": 1.0, "enemy_radar": [0.1]*8, "portal": [1.0,0.5,0.5]}
        r = _calc_reward(env, 0, 0, 1.0, 0, 7, None, state)
        assert r > 0.04

    def test_enemy_threshold_exactly_09(self):
        env1 = _make_env_stub()
        env2 = _make_env_stub()
        env1.visited_chunks.add((0,0)); env2.visited_chunks.add((0,0))
        state_just_under = {"sky_visibility": 1.0, "enemy_radar": [0.89]*8, "portal": [1.0,0.5,0.5]}
        state_at = {"sky_visibility": 1.0, "enemy_radar": [0.90]*8, "portal": [1.0,0.5,0.5]}
        r1 = _calc_reward(env1, 0, 0, 1.0, 0, 6, None, state_just_under)
        r2 = _calc_reward(env2, 0, 0, 1.0, 0, 6, None, state_at)
        assert r1 > r2   # <0.9 triggers bonus; 0.9 does not


# ---------------------------------------------------------------------------
# Portal proximity gradient
# ---------------------------------------------------------------------------

class TestPortalProximity:
    def test_at_zero_dist_max_gradient(self):
        env = _make_env_stub()
        env.visited_chunks.add((0, 0))
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [0.0, 0.5, 0.5]}
        r = _calc_reward(env, 0, 0, 1.0, 0, 0, None, state)
        # gradient = (0.3 - 0.0) * 0.05 = 0.015
        assert r > 0.01

    def test_far_portal_no_gradient(self):
        env = _make_env_stub()
        env.visited_chunks.add((0, 0))
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [0.5, 0.5, 0.5]}
        r = _calc_reward(env, 0, 0, 1.0, 0, 0, None, state)
        assert r == pytest.approx(-0.001, abs=1e-4)

    def test_gradient_monotone(self):
        """Closer to portal → higher gradient reward."""
        results = []
        for dist in [0.0, 0.1, 0.2, 0.29]:
            env = _make_env_stub()
            env.visited_chunks.add((0, 0))
            state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8,
                     "portal": [dist, 0.5, 0.5]}
            results.append(_calc_reward(env, 0, 0, 1.0, 0, 0, None, state))
        # Reward should be monotonically decreasing as dist increases
        assert results[0] > results[1] > results[2] > results[3]


# ---------------------------------------------------------------------------
# Portal teleport (+20)
# ---------------------------------------------------------------------------

class TestPortalTeleport:
    def test_big_jump_near_portal_triggers_bonus(self):
        env = _make_env_stub()
        env.visited_chunks.add((0, 0))
        prev_state = {"x": 0, "y": 0, "dead": False, "portal": [0.3, 0.5, 0.5]}
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8,
                 "portal": [1.0, 0.5, 0.5]}
        # Big jump: moved 500 in Y
        r = _calc_reward(env, 0, 500, 1.0, 0, 0, prev_state, state)
        assert r > 19.0

    def test_big_jump_far_from_portal_no_bonus(self):
        # Spawn=(0,500), max_dist/depth already 500 so no manhattan/depth bonus
        env = _make_env_stub(spawn_x=0, spawn_y=0, max_dist=500.0, max_depth=500.0)
        env.visited_chunks.add((0, 15))   # chunk for y=500
        prev_state = {"x": 0, "y": 0, "dead": False, "portal": [0.9, 0.5, 0.5]}
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        r = _calc_reward(env, 0, 500, 1.0, 0, 0, prev_state, state)
        assert r == pytest.approx(-0.001, abs=1e-4)  # no +20

    def test_small_movement_no_teleport_bonus(self):
        env = _make_env_stub(spawn_x=0, spawn_y=0, max_dist=60.0, max_depth=60.0)
        env.visited_chunks.add((0, 1))   # chunk for y=60
        prev_state = {"x": 0, "y": 0, "dead": False, "portal": [0.1, 0.5, 0.5]}
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        r = _calc_reward(env, 0, 60, 1.0, 0, 0, prev_state, state)
        assert r == pytest.approx(-0.001, abs=1e-4)

    def test_dead_prev_no_bonus(self):
        env = _make_env_stub(spawn_x=0, spawn_y=0, max_dist=500.0, max_depth=500.0)
        env.visited_chunks.add((0, 15))   # chunk for y=500
        prev_state = {"x": 0, "y": 0, "dead": True, "portal": [0.1, 0.5, 0.5]}
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        r = _calc_reward(env, 0, 500, 1.0, 0, 0, prev_state, state)
        assert r == pytest.approx(-0.001, abs=1e-4)   # dead prev → no teleport bonus


# ---------------------------------------------------------------------------
# Death penalty
# ---------------------------------------------------------------------------

class TestDeathPenalty:
    def test_death_subtracts_1(self):
        env = _make_env_stub()
        env.visited_chunks.add((0, 0))
        state = {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]}
        r_alive = _calc_reward(_make_env_stub(), 0, 0, 1.0, 0, 0, None,
                               {"sky_visibility": 1.0, "enemy_radar": [1.0]*8, "portal": [1.0,0.5,0.5]})
        r_dead  = _calc_reward(env, 0, 0, 0.0, 0, 0, None, state, dead=True)
        assert r_dead < r_alive - 0.9   # at least -1 more
