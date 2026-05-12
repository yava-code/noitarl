"""Shared fixtures and sys.path setup for NoitaRL tests."""
import sys
import os

import pytest

# Project root on sys.path so imports work without installing the package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Config isolation — clear env vars that Config reads so .env files in the
# project directory don't silently override expected defaults in tests.
# ---------------------------------------------------------------------------
_CONFIG_ENV_VARS = [
    "NOITA_HOST", "NOITA_BASE_PORT", "N_ENVS",
    "TOTAL_TIMESTEPS", "RUN_NAME", "RESUME_FROM",
    "LEARNING_RATE", "N_STEPS", "BATCH_SIZE", "N_EPOCHS",
    "GAMMA", "GAE_LAMBDA", "CLIP_RANGE", "ENT_COEF", "VF_COEF", "MAX_GRAD_NORM",
    "CHECKPOINT_DIR", "CHECKPOINT_FREQ",
    "LOG_DIR", "TENSORBOARD_DIR", "LOG_LEVEL",
    "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "TELEGRAM_NOTIFY_EVERY",
    "GROQ_API_KEY",
    "WANDB_ENABLED", "WANDB_PROJECT", "WANDB_ENTITY",
]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Remove all Config-relevant env vars so defaults are predictable."""
    for var in _CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
