import os
import pytest
from unittest.mock import patch, MagicMock

from config import Config
from train import setup_logging, setup_wandb

@pytest.fixture
def mock_config():
    cfg = MagicMock(spec=Config)
    cfg.checkpoint_dir = "/tmp/checkpoints"
    cfg.tensorboard_dir = "/tmp/tb"
    cfg.log_level = "INFO"
    cfg.log_dir = "/tmp/logs"
    cfg.wandb_project = "noita-rl"
    cfg.wandb_entity = "jules"
    cfg.wandb_enabled = True
    cfg.n_envs = 1
    cfg.total_timesteps = 1000
    cfg.learning_rate = 0.001
    cfg.n_steps = 2048
    cfg.batch_size = 64
    cfg.n_epochs = 10
    cfg.gamma = 0.99
    cfg.ent_coef = 0.01
    return cfg

def test_setup_logging(tmp_path, mock_config):
    mock_config.checkpoint_dir = str(tmp_path / "checkpoints")
    mock_config.tensorboard_dir = str(tmp_path / "tb")

    with patch("train.logger") as mock_logger, patch("sys.stderr"):
        setup_logging(mock_config, "test_run")

        # Verify logger.add was called
        assert mock_logger.add.call_count > 0

def test_setup_wandb_enabled(mock_config):
    with patch("train.wandb.init", create=True) as mock_init:
        setup_wandb(mock_config, "test_run")
        mock_init.assert_called_once()

def test_setup_wandb_disabled(mock_config):
    mock_config.wandb_enabled = False
    with patch("train.wandb.init", create=True) as mock_init:
        setup_wandb(mock_config, "test_run")
        mock_init.assert_not_called()

def test_setup_wandb_error(mock_config):
    with patch("train.wandb.init", side_effect=Exception("WandB Error"), create=True) as mock_init, \
         patch("train.logger.warning") as mock_warn:
        setup_wandb(mock_config, "test_run")
        mock_warn.assert_called_once()
