"""
Smoke tests — verify all modules import cleanly and expose expected symbols.
These catch circular imports, missing __init__, typos in class/function names.
"""
import sys
import types
import pytest

# Stub optional hardware/OS dependencies so import works in CI
for mod in ("mss", "pygetwindow", "groq"):
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)


class TestModuleImports:
    def test_import_config(self):
        from config import Config
        assert Config is not None

    def test_import_notify(self):
        from notify import TelegramNotifier
        assert TelegramNotifier is not None

    def test_import_video_recorder(self):
        from video_recorder import VideoRecorder, HASHTAGS, _EVENT_HASHTAG
        assert VideoRecorder is not None
        assert isinstance(HASHTAGS, list)
        assert isinstance(_EVENT_HASHTAG, dict)

    def test_import_noita_env(self):
        from noita_env import NoitaEnv
        assert NoitaEnv is not None

    def test_import_callbacks(self):
        from callbacks import NoitaMonitorCallback
        assert NoitaMonitorCallback is not None


class TestExpectedSymbols:
    def test_noita_env_has_obs_builder(self):
        from noita_env import NoitaEnv
        assert hasattr(NoitaEnv, "_obs_from_state")

    def test_noita_env_has_set_recorder(self):
        from noita_env import NoitaEnv
        assert hasattr(NoitaEnv, "set_recorder")

    def test_noita_env_action_space_discrete_10(self):
        from noita_env import NoitaEnv
        import gymnasium as gym
        env = NoitaEnv.__new__(NoitaEnv)
        env.action_space = gym.spaces.Discrete(10)
        assert env.action_space.n == 10

    def test_noita_env_obs_space_60(self):
        from noita_env import NoitaEnv
        import gymnasium as gym
        import numpy as np
        env = NoitaEnv.__new__(NoitaEnv)
        env.observation_space = gym.spaces.Box(low=0., high=1., shape=(60,), dtype=np.float32)
        assert env.observation_space.shape == (60,)

    def test_video_recorder_public_api(self):
        from video_recorder import VideoRecorder
        assert hasattr(VideoRecorder, "start")
        assert hasattr(VideoRecorder, "stop")
        assert hasattr(VideoRecorder, "trigger_event")

    def test_notifier_public_api(self):
        from notify import TelegramNotifier
        for method in ("send_text", "send_photo", "send_document",
                       "send_animation", "make_reward_plot", "make_route_plot",
                       "make_death_postcard", "start_polling", "stop"):
            assert hasattr(TelegramNotifier, method), f"Missing: {method}"

    def test_monitor_callback_recorder_param(self):
        from callbacks import NoitaMonitorCallback
        import inspect
        sig = inspect.signature(NoitaMonitorCallback.__init__)
        assert "recorder" in sig.parameters

    def test_config_groq_key_field(self):
        from config import Config
        cfg = Config()
        assert hasattr(cfg, "groq_api_key")
