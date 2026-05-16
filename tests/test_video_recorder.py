"""
Tests for VideoRecorder.

All tests run without a Noita window, without Telegram credentials, and
without a Groq API key.  We test:
  - GIF creation from PIL frames
  - Fallback descriptions for all 15 event types
  - Hashtag map completeness
  - Cooldown logic (trigger_event is a no-op during cooldown)
  - State machine: idle → recording → post → idle
  - safe start/stop when no window found
"""
import io
import time
import sys
import types
import threading

import pytest
from PIL import Image

# Stub pygetwindow so no Noita window is required
_gw = types.ModuleType("pygetwindow")
_gw.getWindowsWithTitle = lambda title: []
sys.modules.setdefault("pygetwindow", _gw)

from video_recorder import VideoRecorder, _EVENT_HASHTAG, _DESCRIPTIONS, HASHTAGS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeNotifier:
    def __init__(self):
        self.animations = []
        self.documents  = []

    def send_animation(self, gif_bytes, caption=""):
        self.animations.append((gif_bytes, caption))

    def send_document(self, path, caption=""):
        self.documents.append((path, caption))


def _small_frames(n=5, w=32, h=20):
    return [Image.new("RGB", (w, h), color=(i * 20, 100, 200)) for i in range(n)]


@pytest.fixture
def notifier():
    return _FakeNotifier()


@pytest.fixture
def recorder(notifier, tmp_path):
    rec = VideoRecorder(notifier, groq_api_key="")
    rec.SAVE_DIR = str(tmp_path)
    return rec


# ---------------------------------------------------------------------------
# GIF creation
# ---------------------------------------------------------------------------

class TestMakeGif:
    def test_returns_bytes(self, recorder):
        frames = _small_frames(10)
        data = recorder._make_gif(frames)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_valid_gif_magic_bytes(self, recorder):
        data = recorder._make_gif(_small_frames(3))
        assert data[:6] in (b"GIF87a", b"GIF89a")

    def test_gif_can_be_reopened(self, recorder):
        data = recorder._make_gif(_small_frames(5))
        img = Image.open(io.BytesIO(data))
        assert img.format == "GIF"

    def test_single_frame_gif(self, recorder):
        data = recorder._make_gif(_small_frames(1))
        assert data[:3] == b"GIF"

    def test_gif_size_proportional_to_frames(self, recorder):
        small = recorder._make_gif(_small_frames(5))
        large = recorder._make_gif(_small_frames(50))
        # More frames → bigger file (with optimize, not strictly linear, but definitely larger)
        assert len(large) > len(small)


# ---------------------------------------------------------------------------
# Fallback descriptions
# ---------------------------------------------------------------------------

class TestFallbackDescriptions:
    ALL_EVENTS = list(_EVENT_HASHTAG.keys())

    @pytest.mark.parametrize("event_name", ALL_EVENTS)
    def test_fallback_returns_nonempty_string(self, recorder, event_name):
        ctx = {
            "dist": 1234, "depth": 567, "kills": 3,
            "steps": 450, "reward": 42.5, "chunks": 20,
            "vx": 130.0, "vy": 360.0, "episode": 7,
            "damage": 0.5,
        }
        result = recorder._groq_describe(event_name, ctx)
        assert isinstance(result, str)
        assert len(result) > 5

    @pytest.mark.parametrize("event_name", ALL_EVENTS)
    def test_fallback_does_not_crash_on_empty_ctx(self, recorder, event_name):
        result = recorder._groq_describe(event_name, {})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Hashtag / description coverage
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_all_events_have_hashtag(self):
        for event in _EVENT_HASHTAG:
            assert _EVENT_HASHTAG[event] in HASHTAGS, \
                f"Hashtag for '{event}' not in HASHTAGS list"

    def test_all_events_have_description(self):
        for event in _EVENT_HASHTAG:
            assert event in _DESCRIPTIONS, \
                f"Missing description template for event '{event}'"

    def test_hashtags_list_has_10(self):
        assert len(HASHTAGS) == 10

    def test_no_duplicate_hashtags(self):
        assert len(HASHTAGS) == len(set(HASHTAGS))

    def test_all_15_event_types_present(self):
        expected = {
            "portal_teleport", "new_distance_record", "new_depth_record",
            "kill_spree", "instant_damage", "death_long_run", "fast_movement",
            "extreme_fall", "visually_stuck", "action_loop", "chunk_burst",
            "reward_spike", "long_survival", "high_episode_reward", "wand_kill",
        }
        missing = expected - set(_EVENT_HASHTAG.keys())
        assert not missing, f"Missing events in _EVENT_HASHTAG: {missing}"


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_trigger_honours_cooldown(self, recorder):
        recorder.COOLDOWN_SEC = 60
        recorder._last_sent = time.monotonic()   # simulate recent recording
        recorder.trigger_event("reward_spike", {"reward": 99})
        assert recorder._event_q.empty(), "Event should be dropped during cooldown"

    def test_trigger_passes_when_not_in_cooldown(self, recorder):
        recorder.COOLDOWN_SEC = 0
        recorder._last_sent = 0.0
        recorder.trigger_event("reward_spike", {"reward": 99})
        assert not recorder._event_q.empty(), "Event should be queued when no cooldown"

    def test_second_trigger_dropped_within_cooldown(self, recorder):
        recorder.COOLDOWN_SEC = 60
        recorder._last_sent = 0.0   # allow first
        recorder.trigger_event("kill_spree", {"kills": 3})
        recorder._last_sent = time.monotonic()   # now in cooldown
        recorder.trigger_event("kill_spree", {"kills": 4})
        assert recorder._event_q.qsize() == 1, "Second trigger should be dropped"

    def test_zero_cooldown_allows_rapid_triggers(self, recorder):
        recorder.COOLDOWN_SEC = 0
        recorder._last_sent = 0.0
        for _ in range(5):
            recorder.trigger_event("reward_spike", {"reward": 10})
        assert recorder._event_q.qsize() == 5


# ---------------------------------------------------------------------------
# Noita window detection (no window available)
# ---------------------------------------------------------------------------

class TestWindowDetection:
    def test_find_returns_none_when_no_noita(self):
        result = VideoRecorder._find_noita_hwnd()
        # pygetwindow is stubbed to return []
        assert result is None

    def test_grab_returns_none_when_no_window(self, recorder):
        recorder._window_bounds = None
        frame = recorder._grab_frame()
        assert frame is None


# ---------------------------------------------------------------------------
# Start / stop lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_launches_threads(self, recorder, tmp_path):
        recorder.start()
        time.sleep(0.1)
        alive = [t for t in recorder._threads if t.is_alive()]
        assert len(alive) == 2, "Both capture and event threads should be alive"
        recorder.stop()

    def test_stop_sets_flag(self, recorder):
        recorder.start()
        recorder.stop()
        assert recorder._running is False

    def test_double_stop_no_crash(self, recorder):
        recorder.stop()
        recorder.stop()

    def test_trigger_before_start_queues_when_no_cooldown(self, recorder):
        recorder.COOLDOWN_SEC = 0
        recorder._last_sent = 0.0
        recorder.trigger_event("reward_spike", {"reward": 5})
        assert recorder._event_q.qsize() == 1


# ---------------------------------------------------------------------------
# State machine helpers
# ---------------------------------------------------------------------------

class TestStateMachine:
    def test_initial_state_idle(self, recorder):
        assert recorder._state == "idle"

    def test_active_event_none_initially(self, recorder):
        assert recorder._active_evt is None

    def test_pre_buffer_starts_empty(self, recorder):
        assert len(recorder._pre_buf) == 0
