from unittest.mock import patch
"""
Tests for TelegramNotifier.

Only tests that don't require a real Telegram token — the notifier runs
in "disabled" mode (empty token) and every method should be a safe no-op.
We also test the static utility methods (plot generation, postcard).
"""
import io
import pytest
from PIL import Image

from notify import TelegramNotifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def noop():
    """Notifier with empty credentials — all sends silently no-op."""
    return TelegramNotifier("", "")


@pytest.fixture
def fake():
    """Notifier with fake but non-empty credentials (won't actually send)."""
    return TelegramNotifier("fake_token_12345", "999888777")


# ---------------------------------------------------------------------------
# No-op behaviour
# ---------------------------------------------------------------------------

class TestNoOp:
    def test_send_text_returns_false(self, noop):
        assert noop.send_text("hello") is False

    def test_send_photo_returns_false(self, noop):
        assert noop.send_photo(b"fakepng") is False

    def test_send_document_returns_false(self, noop, tmp_path):
        f = tmp_path / "dummy.zip"
        f.write_bytes(b"data")
        assert noop.send_document(str(f)) is False

    def test_send_animation_returns_false(self, noop):
        assert noop.send_animation(b"fakegif") is False

    def test_start_polling_no_crash(self, noop):
        noop.start_polling()   # should silently return
        noop.stop()

    def test_enabled_flag_false(self, noop):
        assert noop._enabled is False

    def test_enabled_flag_true(self, fake):
        assert fake._enabled is True


# ---------------------------------------------------------------------------
# Reward plot
# ---------------------------------------------------------------------------

class TestRewardPlot:
    def test_returns_bytes(self):
        rewards = [float(i) for i in range(50)]
        data = TelegramNotifier.make_reward_plot(rewards)
        assert isinstance(data, bytes)
        assert len(data) > 100

    def test_valid_png_header(self):
        data = TelegramNotifier.make_reward_plot([1.0, 2.0, 3.0])
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_single_episode_no_crash(self):
        data = TelegramNotifier.make_reward_plot([42.0])
        assert isinstance(data, bytes)

    def test_empty_rewards_no_crash(self):
        data = TelegramNotifier.make_reward_plot([])
        assert isinstance(data, bytes)

    def test_custom_title(self):
        data = TelegramNotifier.make_reward_plot([1.0]*10, title="Custom Title")
        assert isinstance(data, bytes)

    def test_moving_average_with_20_points(self):
        data = TelegramNotifier.make_reward_plot(list(range(20)))
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_negative_rewards(self):
        data = TelegramNotifier.make_reward_plot([-5.0, -3.0, -1.0, 0.5])
        assert isinstance(data, bytes)


# ---------------------------------------------------------------------------
# Route plot
# ---------------------------------------------------------------------------

class TestRoutePlot:
    def test_returns_bytes(self):
        xs = list(range(100))
        ys = [x * 2 for x in xs]
        data = TelegramNotifier.make_route_plot(xs, ys)
        assert isinstance(data, bytes)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_single_point(self):
        data = TelegramNotifier.make_route_plot([0.0], [0.0])
        assert isinstance(data, bytes)

    def test_empty_route(self):
        data = TelegramNotifier.make_route_plot([], [])
        assert isinstance(data, bytes)

    def test_custom_title(self):
        data = TelegramNotifier.make_route_plot([0,1], [0,1], title="My Route")
        assert isinstance(data, bytes)


# ---------------------------------------------------------------------------
# Death postcard
# ---------------------------------------------------------------------------

class TestDeathPostcard:
    def _make_png(self):
        img = Image.new("RGB", (200, 100), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_empty_bytes_returns_empty(self):
        result = TelegramNotifier.make_death_postcard(b"", "stats")
        assert result == b""

    def test_returns_bytes_for_valid_png(self):
        result = TelegramNotifier.make_death_postcard(self._make_png(), "ep 42")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_output_is_valid_png(self):
        result = TelegramNotifier.make_death_postcard(self._make_png(), "test")
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_empty_stats_text_no_crash(self):
        result = TelegramNotifier.make_death_postcard(self._make_png(), "")
        assert isinstance(result, bytes)

    def test_long_stats_text_no_crash(self):
        result = TelegramNotifier.make_death_postcard(
            self._make_png(), "a" * 200
        )
        assert isinstance(result, bytes)

    def test_invalid_png_returns_gracefully(self):
        result = TelegramNotifier.make_death_postcard(b"not a png at all", "stats")
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# Noita window capture (no window available)
# ---------------------------------------------------------------------------

class TestScreenCapture:
    def test_returns_bytes_or_empty(self, noop):
        result = noop.capture_noita_screen()
        assert isinstance(result, bytes)

    def test_overlay_text_no_crash(self, noop):
        result = noop.capture_noita_screen(overlay_text="test overlay")
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# Register / stats provider
# ---------------------------------------------------------------------------

class TestCommandRegistration:
    def test_register_command_stores(self, noop):
        noop.register_command("mytest", lambda: None, "test")
        assert "/mytest" in noop._handlers

    def test_register_stats_provider(self, noop):
        noop.register_stats_provider(lambda: "stats text")
        assert noop._stats_fn is not None
        assert noop._stats_fn() == "stats text"

# ---------------------------------------------------------------------------
# AI Status
# ---------------------------------------------------------------------------

class TestAIStatus:
    def test_empty_groq_key(self, noop):
        result = noop.generate_ai_status(groq_key="", stats_context="Test stats")
        assert result == "⚠️ Groq API key not set. Please set GROQ_API_KEY in .env."

    def test_empty_groq_key_none(self, noop):
        result = noop.generate_ai_status(groq_key=None, stats_context="Test stats")
        assert result == "⚠️ Groq API key not set. Please set GROQ_API_KEY in .env."

class TestSetupBotMenu:
    def test_setup_bot_menu_error(self, fake):
        with patch('requests.post', side_effect=Exception('Mock API Error')):
            # Should not crash
            fake.setup_bot_menu()
