"""Smoke tests for the behavioural-cloning pipeline.

These tests avoid the heavy dependencies (pynput, Noita) by importing the
specific helpers we need. They cover:
- mouse_to_wand binning for the 8 cardinal/intercardinal directions
- compose_action mapping (a couple of representative key combinations)
- BCStubEnv space shapes match NoitaEnv
- _load_bc_warmstart skips value_net keys

Run with:  pytest tests/test_bc.py -q
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest


# Stub heavy / unavailable hardware modules before importing the project.
# IMPORTANT: do NOT stub cv2 — SB3.atari_wrappers does cv2.ocl.setUseOpenCL()
# at import time, which crashes on a stub. cv2 is a real install dep.
for _mod in ("mss", "pygetwindow", "win32gui", "win32process", "win32api",
             "win32con", "pynput", "pynput.keyboard", "pynput.mouse"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# Loguru shim — record_human.py imports `loguru.logger` at top level.
import loguru   # type: ignore  # noqa
if not hasattr(loguru, "logger"):
    class _NopLogger:
        def __getattr__(self, _):
            return lambda *a, **k: None
    loguru.logger = _NopLogger()  # type: ignore[attr-defined]


PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))


# ── mouse_to_wand: 8 directions ──────────────────────────────────────────────
@pytest.mark.parametrize("dx,dy,expected_bin", [
    ( 100,    0, 2),   # R
    ( 100, -100, 3),   # UR
    (   0, -100, 4),   # U
    (-100, -100, 5),   # UL
    (-100,    0, 6),   # L
    (-100,  100, 7),   # DL
    (   0,  100, 8),   # D
    ( 100,  100, 9),   # DR
])
def test_mouse_to_wand_cardinals(dx, dy, expected_bin):
    # Skip the test if pynput stub is missing real attrs we don't need.
    # record_human imports pynput.keyboard.Key etc. Provide stand-ins.
    kb = sys.modules["pynput.keyboard"]
    if not hasattr(kb, "Key"):
        class _Key:
            esc = type("E", (), {"name": "esc"})()
        kb.Key = _Key
    if not hasattr(kb, "Listener"):
        kb.Listener = type("L", (), {"__init__": lambda *a, **k: None,
                                     "start": lambda self: None,
                                     "stop":  lambda self: None})
    ms = sys.modules["pynput.mouse"]
    if not hasattr(ms, "Listener"):
        ms.Listener = type("L", (), {"__init__": lambda *a, **k: None,
                                     "start": lambda self: None,
                                     "stop":  lambda self: None})
    # noita_env imports — stub away anything that touches Windows libs.
    from record_human import mouse_to_wand
    assert mouse_to_wand(dx, dy) == expected_bin


def test_mouse_to_wand_dead_zone_defaults_right():
    from record_human import mouse_to_wand
    # |d| < 8 px → ambiguous → default Right (2). Caller is responsible for
    # gating on whether the mouse is actually clicked.
    assert mouse_to_wand(1.0, 0.0) == 2
    assert mouse_to_wand(0.0, 0.0) == 2


# ── compose_action: representative key combos ────────────────────────────────
def test_compose_action_idle():
    from record_human import compose_action
    a = compose_action(set(), set(), (0, 0), (100.0, 100.0))
    assert a.tolist() == [0, 0, 0, 0, 0]


def test_compose_action_right_jump_fire_up():
    from record_human import compose_action
    a = compose_action(
        state_keys={"d", "w"},
        state_buttons={"left"},
        mouse_xy=(500, 0),        # well above centre → wand=4 (U) if dx≈0,dy<0
        window_center=(500, 500),
    )
    assert a[0] == 2  # right
    assert a[1] == 1  # jump
    assert a[4] == 4  # wand U


def test_compose_action_jetpack_via_rmb():
    from record_human import compose_action
    a = compose_action(set(), {"right"}, (0, 0), (100.0, 100.0))
    assert a[2] == 1  # jetpack on
    assert a[4] == 0  # no fire


def test_compose_action_kick_via_f():
    from record_human import compose_action
    a = compose_action({"f"}, set(), (0, 0), (100.0, 100.0))
    assert a[3] == 1


# ── Layout-independent VK detection (regression for Russian-layout bug) ──────
def test_key_name_uses_vk_for_letter_keys_under_cyrillic_layout():
    """When a non-Latin keyboard layout is active, pynput's `.char` returns
    the translated symbol (e.g. 'в' instead of 'd'). The _key_name helper
    must fall back to the Windows VK code so WASD detection survives.
    Regression test for the "agent walks at 1 px/sec" bug reported by the
    user on 2026-05-17."""
    from record_human import _key_name

    class _StubKeyCode:
        def __init__(self, vk: int, char: str):
            self.vk = vk
            self.char = char

    # Russian D physical key → char='в' (Cyrillic ve) → should still map to 'd'.
    assert _key_name(_StubKeyCode(0x44, "в")) == "d"
    # English A physical key with US layout → char='a' — also maps to 'a' via
    # the VK lookup (not the char fallback), since 0x41 is in _VK_PHYSICAL.
    assert _key_name(_StubKeyCode(0x41, "a")) == "a"
    # An unmapped key (e.g. '7') with no entry in _VK_PHYSICAL should fall
    # back to the .char path.
    assert _key_name(_StubKeyCode(0x37, "7")) == "7"


def test_stop_key_is_f10_not_esc():
    """The recorder stops on F10 only — ESC is a Noita menu key and would
    cause silent recorder termination if accidentally pressed. Hard-coded
    here so a future "let me make ESC stop again" change can't slip in
    unnoticed."""
    from record_human import HumanInputState
    assert HumanInputState.STOP_KEY == "f10"


# ── BCStubEnv spaces ─────────────────────────────────────────────────────────
def test_bcstub_env_spaces_match_real_env():
    """BCStubEnv must expose the same observation/action spaces as NoitaEnv so
    SB3 builds a shape-compatible MultiInputPolicy that can later receive the
    state_dict via _load_bc_warmstart."""
    import gymnasium as gym
    from train_bc import BCStubEnv

    env = BCStubEnv(cv_enabled=True, image_size=84, frame_stack=4)
    assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
    assert env.action_space.nvec.tolist() == [3, 2, 2, 2, 10]
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert env.observation_space["image"].shape == (5, 84, 84)
    assert env.observation_space["sensors"].shape == (60,)


# ── _load_bc_warmstart filtering logic ───────────────────────────────────────
def test_load_bc_warmstart_skips_value_net_keys():
    """Just exercises the key-filter logic — does NOT load real PPO weights."""
    import torch
    from train import _load_bc_warmstart

    class _FakePolicy:
        def __init__(self):
            self._sd = {
                "features_extractor.cnn.0.weight": torch.zeros(32, 5, 8, 8),
                "mlp_extractor.policy_net.0.weight": torch.zeros(256, 320),
                "mlp_extractor.value_net.0.weight":  torch.zeros(256, 320),
                "value_net.weight":                  torch.zeros(1, 128),
                "action_net.weight":                 torch.zeros(19, 128),
            }
        def state_dict(self): return dict(self._sd)
        def load_state_dict(self, sd):
            for k, v in sd.items():
                assert k in self._sd
                self._sd[k] = v

    class _FakeModel:
        device = "cpu"
        policy = _FakePolicy()

    fake_path = Path(__file__).resolve().parent / "_bc_smoke.pth"
    # BC checkpoint deliberately INCLUDES value_net keys with garbage; the
    # loader must refuse to copy them through.
    bc_sd = {
        "features_extractor.cnn.0.weight":   torch.ones(32, 5, 8, 8),
        "mlp_extractor.policy_net.0.weight": torch.ones(256, 320),
        "mlp_extractor.value_net.0.weight":  torch.ones(256, 320) * 999,
        "value_net.weight":                  torch.ones(1, 128) * 999,
        "action_net.weight":                 torch.ones(19, 128),
        "spurious_unknown_key":               torch.ones(4),
    }
    torch.save(bc_sd, fake_path)

    try:
        model = _FakeModel()
        _load_bc_warmstart(model, str(fake_path))
        sd = model.policy.state_dict()
        # Loaded ones (1.0)
        assert torch.all(sd["features_extractor.cnn.0.weight"] == 1.0)
        assert torch.all(sd["mlp_extractor.policy_net.0.weight"] == 1.0)
        assert torch.all(sd["action_net.weight"] == 1.0)
        # Skipped: value_net keys must remain zero (the fresh PPO init).
        assert torch.all(sd["mlp_extractor.value_net.0.weight"] == 0.0)
        assert torch.all(sd["value_net.weight"] == 0.0)
    finally:
        try: fake_path.unlink()
        except FileNotFoundError: pass
