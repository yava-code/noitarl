"""
Tests for NoitaEnv._obs_from_state() — the observation vector builder.

All tests are pure-Python and require no Noita process, no WebSocket,
no mss, and no pygetwindow.  We instantiate the class via __new__ to
skip __init__ (which starts the WS server and tries to import hardware libs).
"""
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env():
    """Return a NoitaEnv instance without running __init__."""
    # Lazy import so the test doesn't fail at collection time if mss is missing
    import importlib, types, sys

    # Stub out hardware-dependent top-level imports that run at module load
    for mod_name in ("mss", "pygetwindow"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    from noita_env import NoitaEnv
    env = NoitaEnv.__new__(NoitaEnv)
    env._recorder = None
    return env


@pytest.fixture
def env():
    return make_env()


FULL_STATE = {
    "rays":             [1.0] * 16,
    "enemy_radar":      [1.0] * 8,
    "liquid_sensors":   [0.0] * 8,
    "projectile_radar": [1.0] * 8,
    "gold_radar":       [1.0] * 8,
    "hp":               1.0,
    "vx":               0.0,
    "vy":               0.0,
    "on_ground":        True,
    "jetpack_fuel":     1.0,
    "wand_ready":       1.0,
    "is_on_fire":       0.0,
    "is_poisoned":      0.0,
    "sky_visibility":   0.0,
    "portal":           [1.0, 0.5, 0.5],
}


# ---------------------------------------------------------------------------
# Shape / dtype
# ---------------------------------------------------------------------------

class TestShape:
    def test_none_state_returns_zeros_60(self, env):
        obs = env._obs_from_state(None)
        assert obs.shape == (60,)
        assert obs.dtype == np.float32
        assert np.all(obs == 0.0)

    def test_full_state_returns_60(self, env):
        obs = env._obs_from_state(FULL_STATE)
        assert obs.shape == (60,)
        assert obs.dtype == np.float32

    def test_all_values_in_range(self, env):
        obs = env._obs_from_state(FULL_STATE)
        assert float(obs.min()) >= 0.0
        assert float(obs.max()) <= 1.0


# ---------------------------------------------------------------------------
# Ray slice  [0..15]
# ---------------------------------------------------------------------------

class TestRays:
    def test_ray_values_passed_through(self, env):
        state = {**FULL_STATE, "rays": [0.5] * 16}
        obs = env._obs_from_state(state)
        np.testing.assert_array_almost_equal(obs[0:16], [0.5] * 16)

    def test_missing_rays_default_to_1(self, env):
        state = {k: v for k, v in FULL_STATE.items() if k != "rays"}
        obs = env._obs_from_state(state)
        np.testing.assert_array_equal(obs[0:16], [1.0] * 16)

    def test_short_rays_padded(self, env):
        state = {**FULL_STATE, "rays": [0.2] * 8}   # only 8 given
        obs = env._obs_from_state(state)
        # float32 vs float64 needs approx comparison
        np.testing.assert_array_almost_equal(obs[0:8], [0.2] * 8, decimal=5)
        np.testing.assert_array_almost_equal(obs[8:16], [1.0] * 8, decimal=5)


# ---------------------------------------------------------------------------
# Radars  [16..47]
# ---------------------------------------------------------------------------

class TestRadars:
    def test_enemy_radar_slice(self, env):
        state = {**FULL_STATE, "enemy_radar": [0.3] * 8}
        obs = env._obs_from_state(state)
        np.testing.assert_array_almost_equal(obs[16:24], [0.3] * 8)

    def test_liquid_sensors_slice(self, env):
        state = {**FULL_STATE, "liquid_sensors": [0.7] * 8}
        obs = env._obs_from_state(state)
        np.testing.assert_array_almost_equal(obs[24:32], [0.7] * 8)

    def test_projectile_radar_slice(self, env):
        state = {**FULL_STATE, "projectile_radar": [0.1] * 8}
        obs = env._obs_from_state(state)
        np.testing.assert_array_almost_equal(obs[32:40], [0.1] * 8)

    def test_gold_radar_slice(self, env):
        state = {**FULL_STATE, "gold_radar": [0.6] * 8}
        obs = env._obs_from_state(state)
        np.testing.assert_array_almost_equal(obs[40:48], [0.6] * 8)

    def test_missing_gold_defaults_to_1(self, env):
        state = {k: v for k, v in FULL_STATE.items() if k != "gold_radar"}
        obs = env._obs_from_state(state)
        np.testing.assert_array_equal(obs[40:48], [1.0] * 8)


# ---------------------------------------------------------------------------
# Scalar features  [48..56]
# ---------------------------------------------------------------------------

class TestScalars:
    def test_hp_at_48(self, env):
        obs = env._obs_from_state({**FULL_STATE, "hp": 0.75})
        assert pytest.approx(obs[48], abs=1e-4) == 0.75

    def test_vx_normalised_at_49(self, env):
        # vx = +200 → normalised = (200/200)*0.5 + 0.5 = 1.0
        obs = env._obs_from_state({**FULL_STATE, "vx": 200.0})
        assert pytest.approx(float(obs[49]), abs=1e-4) == 1.0

    def test_vx_negative(self, env):
        # vx = -200 → 0.0
        obs = env._obs_from_state({**FULL_STATE, "vx": -200.0})
        assert pytest.approx(float(obs[49]), abs=1e-4) == 0.0

    def test_vx_zero_maps_to_half(self, env):
        obs = env._obs_from_state({**FULL_STATE, "vx": 0.0})
        assert pytest.approx(float(obs[49]), abs=1e-4) == 0.5

    def test_vx_clamped_above(self, env):
        obs = env._obs_from_state({**FULL_STATE, "vx": 999.0})
        assert pytest.approx(float(obs[49]), abs=1e-4) == 1.0

    def test_vy_at_50(self, env):
        obs = env._obs_from_state({**FULL_STATE, "vy": 100.0})
        assert 0.0 < float(obs[50]) < 1.0

    def test_on_ground_true(self, env):
        obs = env._obs_from_state({**FULL_STATE, "on_ground": True})
        assert float(obs[51]) == 1.0

    def test_on_ground_false(self, env):
        obs = env._obs_from_state({**FULL_STATE, "on_ground": False})
        assert float(obs[51]) == 0.0

    def test_is_on_fire_at_54(self, env):
        obs = env._obs_from_state({**FULL_STATE, "is_on_fire": 1.0})
        assert float(obs[54]) == 1.0

    def test_is_poisoned_at_55(self, env):
        obs = env._obs_from_state({**FULL_STATE, "is_poisoned": 1.0})
        assert float(obs[55]) == 1.0

    def test_sky_visibility_at_56(self, env):
        obs = env._obs_from_state({**FULL_STATE, "sky_visibility": 0.8})
        assert pytest.approx(float(obs[56]), abs=1e-4) == 0.8


# ---------------------------------------------------------------------------
# Portal features  [57..59]
# ---------------------------------------------------------------------------

class TestPortal:
    def test_portal_default_no_portal(self, env):
        state = {k: v for k, v in FULL_STATE.items() if k != "portal"}
        obs = env._obs_from_state(state)
        assert float(obs[57]) == 1.0   # dist = 1 (nothing nearby)
        assert float(obs[58]) == 0.5   # dx centre
        assert float(obs[59]) == 0.5   # dy centre

    def test_portal_values_passed(self, env):
        state = {**FULL_STATE, "portal": [0.25, 0.8, 0.1]}
        obs = env._obs_from_state(state)
        assert pytest.approx(float(obs[57]), abs=1e-4) == 0.25
        assert pytest.approx(float(obs[58]), abs=1e-4) == 0.8
        assert pytest.approx(float(obs[59]), abs=1e-4) == 0.1

    def test_portal_wrong_type_falls_back(self, env):
        state = {**FULL_STATE, "portal": "bad_value"}
        obs = env._obs_from_state(state)
        assert float(obs[57]) == 1.0

    def test_portal_wrong_length_falls_back(self, env):
        state = {**FULL_STATE, "portal": [0.5, 0.5]}   # only 2 elements
        obs = env._obs_from_state(state)
        assert float(obs[57]) == 1.0

    def test_portal_at_zero_dist(self, env):
        state = {**FULL_STATE, "portal": [0.0, 0.5, 0.5]}
        obs = env._obs_from_state(state)
        assert float(obs[57]) == 0.0
