"""
record_human.py — capture human Noita gameplay for behavioural-cloning warm-start.

Workflow:
    1. Launch Noita with the rl_agent mod, same as for normal training.
    2. Run this script. It connects to the NoitaEnv WebSocket on --port and
       starts a pynput keyboard+mouse listener.
    3. Click into the Noita window and play. Each env.step() captures the
       human's mapped MultiDiscrete action together with the observation that
       triggered it. Data is written to `data/bc_dataset/chunk_NNNNNN.npz`
       every `--chunk-size` steps so RAM stays bounded.
    4. Press ESC to flush and quit.

Input mapping (matches MultiDiscrete([3,2,2,2,10])):
    A / D                       → move      (1=Left, 2=Right, 0=Idle)
    W or Space                  → jump      (1/0)
    Shift (L or R) or RightClick→ jetpack   (1/0)
    F or MiddleClick            → kick      (1/0)
    LeftClick + mouse direction → wand      (0=idle, 2..9 = 8 cardinals)

The mouse direction is dot-product-binned against the same WAND_DIRS lookup
init.lua uses, so the recorded action is numerically identical to what the
agent would output for the same intent.

Pre-req: `pip install pynput`. Add to requirements.txt — bundled here as a
soft dep so the rest of the project still imports without it.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

# Soft dep: pynput is only needed by this script. Fail loudly with install hint.
try:
    from pynput import keyboard, mouse
except ImportError as _e:
    print("ERROR: pynput is required for human recording.", file=sys.stderr)
    print("       pip install pynput", file=sys.stderr)
    raise

try:
    import win32gui
except ImportError:
    win32gui = None

from noita_env import NoitaEnv, _find_noita_hwnd_by_pid, _find_any_noita_hwnd


# Same 8-direction table init.lua uses. Index = wand-head value (2..9).
WAND_DIRS = {
    2: ( 1.0,  0.0),         # R
    3: ( 0.7071, -0.7071),   # UR
    4: ( 0.0, -1.0),         # U
    5: (-0.7071, -0.7071),   # UL
    6: (-1.0,  0.0),         # L
    7: (-0.7071,  0.7071),   # DL
    8: ( 0.0,  1.0),         # D
    9: ( 0.7071,  0.7071),   # DR
}
# Within this radius of window centre, mouse direction is ambiguous.
# Default to wand=2 (R) so we never emit half-snapped intermediate angles.
MOUSE_DEAD_ZONE_PX = 8.0


def mouse_to_wand(dx: float, dy: float) -> int:
    """Map a mouse offset (screen coords, y-down) to a wand bin 2..9.
    Returns 2 (Right) if the mouse is within the dead zone — caller should
    only invoke this when the agent is actually firing."""
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < MOUSE_DEAD_ZONE_PX:
        return 2
    nx, ny = dx / norm, dy / norm
    best_idx, best_dot = 2, -2.0
    for idx, (wx, wy) in WAND_DIRS.items():
        d = nx * wx + ny * wy
        if d > best_dot:
            best_dot, best_idx = d, idx
    return best_idx


class HumanInputState:
    """Thread-safe snapshot of currently-held keys and mouse buttons.

    Updated by pynput listener callbacks; read once per env.step() by the
    main thread. We deliberately keep this dumb (sets of held keys + last
    mouse position) — composing the MultiDiscrete action happens in the
    main loop so it sees a consistent snapshot.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.keys: set = set()          # held keyboard keys (string names)
        self.buttons: set = set()       # held mouse buttons (string names)
        self.mouse_xy: tuple[int, int] = (0, 0)
        self.stop_requested: bool = False

    def snapshot(self) -> tuple[set, set, tuple[int, int]]:
        with self._lock:
            return set(self.keys), set(self.buttons), self.mouse_xy

    # ── pynput callbacks ──────────────────────────────────────────────────
    # Stop key is F10 (not ESC): Noita uses ESC for its pause menu, and
    # pressing it accidentally inside the game would silently terminate a
    # recording session. F10 has no in-game binding so it's safe.
    STOP_KEY = "f10"

    def on_key_press(self, key) -> Optional[bool]:
        name = _key_name(key)
        with self._lock:
            self.keys.add(name)
        if name == self.STOP_KEY:
            with self._lock:
                self.stop_requested = True
            return False  # stops the keyboard listener
        return None

    def on_key_release(self, key) -> None:
        name = _key_name(key)
        with self._lock:
            self.keys.discard(name)

    def on_mouse_move(self, x: int, y: int) -> None:
        with self._lock:
            self.mouse_xy = (int(x), int(y))

    def on_mouse_click(self, x: int, y: int, button, pressed: bool) -> None:
        name = str(button).split(".")[-1]  # e.g. "left", "right", "middle"
        with self._lock:
            self.mouse_xy = (int(x), int(y))
            if pressed:
                self.buttons.add(name)
            else:
                self.buttons.discard(name)


_VK_PHYSICAL = {
    # Windows VK codes for the QWERTY letter positions we care about. Looking
    # these up by VK instead of `.char` makes detection layout-independent:
    # `key.char` returns the layout-translated character (e.g. 'ф' for the A
    # key when Russian layout is active), which would break our WASD matching.
    0x41: "a",   # A
    0x44: "d",   # D
    0x53: "s",   # S
    0x57: "w",   # W
    0x46: "f",   # F
    0x51: "q",   # Q (reserved for future binds; logged for debug)
}


def _key_name(key) -> str:
    """Normalize pynput key objects → lowercase string for compose_action.

    Resolution order:
      1. Special keys (Esc / Shift / Space / F10 …): use `key.name`.
      2. Letter keys: look up `key.vk` (Windows virtual-key code) in
         _VK_PHYSICAL — this is the layout-independent physical-key identity.
      3. Fall back to `key.char` (used for non-Latin keys we haven't bound).
    """
    if isinstance(key, keyboard.Key):
        return key.name.lower()
    vk = getattr(key, "vk", None)
    if vk in _VK_PHYSICAL:
        return _VK_PHYSICAL[vk]
    try:
        ch = key.char
        return ch.lower() if ch else ""
    except AttributeError:
        return ""


def compose_action(
    state_keys: set,
    state_buttons: set,
    mouse_xy: tuple[int, int],
    window_center: Optional[tuple[float, float]],
) -> np.ndarray:
    """Build the 5-element MultiDiscrete action from a snapshot of inputs."""
    # [0] move
    left  = "a" in state_keys
    right = "d" in state_keys
    if left and not right:
        move = 1
    elif right and not left:
        move = 2
    else:
        move = 0

    # [1] jump
    jump = int(("w" in state_keys) or ("space" in state_keys))

    # [2] jetpack
    jetpack = int(
        ("shift" in state_keys) or ("shift_l" in state_keys) or ("shift_r" in state_keys)
        or ("right" in state_buttons)
    )

    # [3] kick
    kick = int(("f" in state_keys) or ("middle" in state_buttons))

    # [4] wand: only when left-click is held AND we know where the window centre is
    if ("left" in state_buttons) and window_center is not None:
        cx, cy = window_center
        dx = mouse_xy[0] - cx
        dy = mouse_xy[1] - cy
        wand = mouse_to_wand(dx, dy)
    else:
        wand = 0

    return np.array([move, jump, jetpack, kick, wand], dtype=np.int64)


class ChunkWriter:
    """Buffers (image, sensors, action) tuples and flushes compressed .npz
    chunks. Images stay uint8 so on-disk size matches the obs space exactly."""

    def __init__(self, out_dir: Path, chunk_size: int, cv_enabled: bool):
        self.out_dir = out_dir
        self.chunk_size = chunk_size
        self.cv_enabled = cv_enabled
        self.images: list[np.ndarray] = []
        self.sensors: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self._next_chunk = self._scan_existing_chunk_idx()
        self.total_steps = 0

    def _scan_existing_chunk_idx(self) -> int:
        """Resume numbering if the user re-runs into the same output dir."""
        existing = sorted(self.out_dir.glob("chunk_*.npz"))
        if not existing:
            return 0
        last = existing[-1].stem  # "chunk_000007"
        try:
            return int(last.split("_")[-1]) + 1
        except ValueError:
            return len(existing)

    def append(self, obs, action: np.ndarray) -> None:
        if self.cv_enabled and isinstance(obs, dict):
            self.images.append(obs["image"].astype(np.uint8, copy=False))
            self.sensors.append(obs["sensors"].astype(np.float32, copy=False))
        else:
            self.sensors.append(obs.astype(np.float32, copy=False))
        self.actions.append(action.astype(np.int64, copy=False))
        self.total_steps += 1
        if len(self.actions) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.actions:
            return
        path = self.out_dir / f"chunk_{self._next_chunk:06d}.npz"
        payload = {
            "sensors": np.stack(self.sensors, axis=0),
            "actions": np.stack(self.actions, axis=0),
        }
        if self.cv_enabled and self.images:
            payload["images"] = np.stack(self.images, axis=0)
        np.savez_compressed(path, **payload)
        logger.info("Flushed chunk {} ({} frames, total {})",
                    path.name, len(self.actions), self.total_steps)
        self.images.clear()
        self.sensors.clear()
        self.actions.clear()
        self._next_chunk += 1


def find_noita_window(pid: Optional[int]) -> Optional[int]:
    if win32gui is None:
        return None
    if pid is not None:
        h = _find_noita_hwnd_by_pid(pid)
        if h is not None:
            return h
    return _find_any_noita_hwnd()


def window_centre(hwnd: int) -> Optional[tuple[float, float]]:
    """Return the screen-coord centre of the Noita client area, or None if
    the window has been minimised/destroyed."""
    if win32gui is None or hwnd is None:
        return None
    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        cw, ch = right - left, bottom - top
        if cw < 8 or ch < 8:
            return None
        sx, sy = win32gui.ClientToScreen(hwnd, (0, 0))
        return (sx + cw / 2.0, sy + ch / 2.0)
    except Exception:
        return None


def _log_heartbeat(total_steps: int, last_action: np.ndarray) -> None:
    """One-line status. Spell the 5 heads out so the user can spot if e.g.
    move is permanently 0 (keyboard layout bug)."""
    m, j, t, k, w = (int(x) for x in last_action)
    logger.info(
        "Recording… {} steps  |  last: move={} jump={} jet={} kick={} wand={}",
        total_steps, m, j, t, k, w,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record human Noita gameplay for BC.")
    p.add_argument("--port", type=int, default=5001,
                   help="WebSocket port that the rl_agent mod connects to (default 5001).")
    p.add_argument("--output", type=str, default="data/bc_dataset",
                   help="Output directory for chunked .npz files.")
    p.add_argument("--chunk-size", type=int, default=1000,
                   help="Number of frames per .npz file (default 1000 ≈ 10 MB compressed).")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Stop after N total steps (0 = unlimited, press ESC to stop).")
    p.add_argument("--noita-pid", type=int, default=None,
                   help="If set, restricts the window-centre lookup to this Noita PID "
                        "(useful when several Noita instances are running).")
    p.add_argument("--dry-run", action="store_true",
                   help="Connect & wire up listeners but write nothing to disk. "
                        "Useful smoke test before a real recording session.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start the env on the same port the Noita mod is configured for. We use
    # the standard NoitaEnv so the (obs, action) pairs we record come from
    # the exact same code path RL training will see at inference time —
    # no train/play distribution shift.
    env = NoitaEnv(port=args.port)
    logger.info("Waiting for Noita to connect on port {}…", args.port)

    state = HumanInputState()
    kb_listener = keyboard.Listener(
        on_press=state.on_key_press, on_release=state.on_key_release)
    ms_listener = mouse.Listener(
        on_move=state.on_mouse_move, on_click=state.on_mouse_click)
    kb_listener.start()
    ms_listener.start()
    logger.info("pynput listeners up. Click into Noita and play.")
    logger.info("Stop key: F10  (do NOT use ESC — Noita uses it for pause).")
    logger.info("Layout-independent: A/D/W/S/F detected by VK code, so "
                "Cyrillic / other non-Latin keyboard layouts work fine.")

    writer = ChunkWriter(out_dir, args.chunk_size, env.cv_enabled)

    hwnd = find_noita_window(args.noita_pid)
    if hwnd is None:
        logger.warning("Couldn't locate Noita window. wand actions will be 0 "
                       "until window detection succeeds (retried each step).")

    try:
        obs, _ = env.reset()
        last_log = time.monotonic()
        while not state.stop_requested:
            # Refresh hwnd lazily — Noita may have only just become visible.
            if hwnd is None:
                hwnd = find_noita_window(args.noita_pid)
            centre = window_centre(hwnd) if hwnd is not None else None

            keys, buttons, mouse_xy = state.snapshot()
            action = compose_action(keys, buttons, mouse_xy, centre)

            if not args.dry_run:
                writer.append(obs, action)

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                # The env resets internally via the Noita-side respawn hack,
                # but we explicitly call reset() to clear stale frame stack.
                obs, _ = env.reset()

            if args.max_steps and writer.total_steps >= args.max_steps:
                logger.info("Reached --max-steps {} — stopping.", args.max_steps)
                break

            # Periodic heartbeat so the user knows the recorder is alive even
            # during long idle stretches. Reports the action distribution for
            # the last window so layout/binding bugs surface immediately
            # (e.g. all-zero moves under a broken keymap).
            now = time.monotonic()
            if now - last_log > 5.0:
                _log_heartbeat(writer.total_steps, action)
                last_log = now
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    finally:
        if not args.dry_run:
            writer.flush()
            logger.info("Wrote {} chunks under {} (total {} steps)",
                        writer._next_chunk, out_dir, writer.total_steps)
        try:
            kb_listener.stop()
        except Exception:
            pass
        try:
            ms_listener.stop()
        except Exception:
            pass
        env.close()


if __name__ == "__main__":
    main()
