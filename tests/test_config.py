"""
Tests for Config — settings loading and derived properties.

We use Config(_env_file=None) throughout to prevent the project's .env file
from overriding code-level defaults during testing.
"""
import os
import pytest
from config import Config


def cfg(**kwargs) -> Config:
    """Return Config with no .env so code-level defaults are authoritative."""
    return Config(_env_file=None, **kwargs)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_port(self):
        assert cfg().noita_base_port == 5001

    def test_default_n_envs(self):
        assert cfg().n_envs == 1

    def test_default_total_timesteps(self):
        assert cfg().total_timesteps == 1_000_000

    def test_default_learning_rate(self):
        assert cfg().learning_rate == pytest.approx(1e-4)

    def test_default_n_steps(self):
        assert cfg().n_steps == 1024

    def test_default_batch_size(self):
        assert cfg().batch_size == 256

    def test_default_n_epochs(self):
        assert cfg().n_epochs == 10

    def test_default_gamma(self):
        assert cfg().gamma == pytest.approx(0.99)

    def test_default_ent_coef(self):
        assert cfg().ent_coef == pytest.approx(0.03)

    def test_default_checkpoint_dir(self):
        assert cfg().checkpoint_dir == "./checkpoints"

    def test_default_tensorboard_dir(self):
        assert cfg().tensorboard_dir == "./noita_ppo_tensorboard"

    def test_default_telegram_empty(self):
        c = cfg()
        assert c.telegram_token == ""
        assert c.telegram_chat_id == ""

    def test_default_groq_api_key_empty(self):
        assert cfg().groq_api_key == ""

    def test_default_wandb_disabled(self):
        assert cfg().wandb_enabled is False

    def test_default_run_name_none(self):
        assert cfg().run_name is None

    def test_default_resume_from_none(self):
        assert cfg().resume_from is None

    def test_log_level_default(self):
        assert cfg().log_level == "INFO"


# ---------------------------------------------------------------------------
# Derived properties
# ---------------------------------------------------------------------------

class TestPorts:
    def test_single_env_one_port(self):
        assert cfg().ports == [5001]

    def test_two_envs_two_ports(self):
        assert cfg(n_envs=2).ports == [5001, 5002]

    def test_three_envs_sequential_ports(self):
        assert cfg(n_envs=3, noita_base_port=5010).ports == [5010, 5011, 5012]

    def test_ports_length_equals_n_envs(self):
        assert len(cfg(n_envs=4).ports) == 4


class TestRunName:
    def test_effective_run_name_contains_prefix(self):
        name = cfg().effective_run_name()
        assert "ppo_1env" in name

    def test_effective_run_name_with_custom(self):
        name = cfg(run_name="my_experiment").effective_run_name()
        assert "my_experiment" in name

    def test_effective_run_name_has_timestamp(self):
        name = cfg().effective_run_name()
        suffix = name.split("_")[-1]
        assert suffix.isdigit()
        assert int(suffix) > 1_700_000_000


class TestTelegramEnabled:
    def test_telegram_disabled_when_empty(self):
        assert cfg().telegram_enabled is False

    def test_telegram_enabled_when_both_set(self):
        assert cfg(telegram_token="tok", telegram_chat_id="123").telegram_enabled is True

    def test_telegram_disabled_missing_chat_id(self):
        assert cfg(telegram_token="tok").telegram_enabled is False


# ---------------------------------------------------------------------------
# Direct field override (no .env or env-var needed)
# ---------------------------------------------------------------------------

class TestFieldOverride:
    def test_learning_rate_override(self):
        assert cfg(learning_rate=0.0005).learning_rate == pytest.approx(0.0005)

    def test_n_steps_override(self):
        assert cfg(n_steps=2048).n_steps == 2048

    def test_groq_api_key_override(self):
        assert cfg(groq_api_key="gsk_test").groq_api_key == "gsk_test"

    def test_total_timesteps_override(self):
        assert cfg(total_timesteps=500_000).total_timesteps == 500_000
