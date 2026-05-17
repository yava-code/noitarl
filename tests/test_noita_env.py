import pytest
from unittest.mock import MagicMock, patch
import asyncio

def make_env():
    import importlib, types, sys

    # Mock problematic modules
    sys.modules['mss'] = MagicMock()
    sys.modules['pygetwindow'] = MagicMock()
    sys.modules['pyrect'] = MagicMock()

    import noita_env
    importlib.reload(noita_env)

    env = noita_env.NoitaEnv.__new__(noita_env.NoitaEnv)
    env._lock = MagicMock()
    env._ws = MagicMock()
    env._sct = MagicMock()
    env._frame_buf = []
    env.port = 1234
    return env

class TestNoitaEnvMethods:
    def test_render_does_not_crash(self):
        env = make_env()
        # Should do nothing and not crash
        env.render()

    def test_close_clears_ws(self):
        env = make_env()
        env._ws = MagicMock()
        assert env._ws is not None
        env.close()
        assert env._ws is None
        env._lock.__enter__.assert_called()

    def test_handle_exception_path(self):
        env = make_env()
        env._ws = None
        # Create a mock socket
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = Exception("Connection closed")

        # Should gracefully exit the loop when exception is caught
        asyncio.run(env._handle(mock_ws))
        # Should set _ws to None on exit
        assert env._ws is None
