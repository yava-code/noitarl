import pytest
from unittest.mock import MagicMock, patch
from callbacks import NoitaMonitorCallback


def test_callbacks_init():
    mock_env = MagicMock()
    mock_env.envs = [MagicMock()]
    mock_env.envs[0].port = 1234

    with patch("callbacks.TelegramNotifier", create=True) as mock_notifier:
        with patch("callbacks.VideoRecorder", create=True) as mock_recorder:
            mock_cfg = MagicMock()
            cb = NoitaMonitorCallback(
                cfg=mock_cfg,
                notifier=mock_notifier,
                verbose=0,
                recorder=mock_recorder
            )
            assert cb is not None
