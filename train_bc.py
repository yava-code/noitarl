"""
train_bc.py — supervised behavioural-cloning warm-start for the Noita PPO policy.

Inputs:
    `data/bc_dataset/chunk_NNNNNN.npz` produced by `record_human.py`.

Output:
    `checkpoints/bc_pretrained.pth` — torch state_dict of the SB3 policy
    (not a full PPO zip, see README in dev_notes/2026-05-22.md for why).

Loss: sum of `F.cross_entropy(logits_head_i, actions[:, i])` across the 5
MultiCategorical heads (move/jump/jetpack/kick/wand). Per-head accuracy is
logged each epoch so the user can spot heads that aren't learning (e.g. wand
collapsing to "always idle" because the dataset is move-heavy).

The PPO model used here is constructed with the SAME `MultiInputPolicy` and
`policy_kwargs` as `train.py` so the resulting state_dict drops straight in
during warm-start.

Usage:
    python train_bc.py
    python train_bc.py --data-dir data/bc_dataset --epochs 10 --batch-size 256
    python train_bc.py --output checkpoints/bc_pretrained.pth --val-split 0.1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Workaround for OpenMP duplicate library error — same fix as train.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from loguru import logger

import gymnasium as gym
from stable_baselines3 import PPO

from config import Config


# ── BC stub environment ──────────────────────────────────────────────────────
# PPO's constructor needs an env to introspect observation/action spaces. We
# don't want to spin up a WebSocket server for that, so we hand it a minimal
# Env whose only contract is that the spaces match the real NoitaEnv exactly.
class BCStubEnv(gym.Env):
    """Dummy env exposing the spaces NoitaEnv uses; never stepped at training."""

    metadata = {"render_modes": []}

    def __init__(self, cv_enabled: bool, image_size: int, frame_stack: int):
        super().__init__()
        self.action_space = gym.spaces.MultiDiscrete([3, 2, 2, 2, 10])
        channels = frame_stack + 1   # matches NoitaEnv._image_channels_total
        if cv_enabled:
            self.observation_space = gym.spaces.Dict({
                "image": gym.spaces.Box(
                    low=0, high=255,
                    shape=(channels, image_size, image_size),
                    dtype=np.uint8,
                ),
                "sensors": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(60,), dtype=np.float32,
                ),
            })
        else:
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(60,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


# ── Dataset ──────────────────────────────────────────────────────────────────
class BCDataset(Dataset):
    """Concatenates all chunked .npz files into RAM-resident arrays.

    Sizing: at 100k frames the sensor + action arrays total ≈ 25 MB; the
    image stack adds ~175 MB at 5×84×84 uint8. Both comfortably fit in RAM
    for any realistic 10-60 min demo. Streaming would only be needed past
    ~1 M frames; out of scope here.
    """

    def __init__(self, chunk_files: list[Path], has_images: bool):
        sensors, actions = [], []
        images = [] if has_images else None
        for f in chunk_files:
            with np.load(f) as data:
                sensors.append(data["sensors"])
                actions.append(data["actions"])
                if has_images:
                    if "images" not in data.files:
                        raise ValueError(
                            f"{f.name} has no 'images' array but cv_enabled=True. "
                            "Re-record with the matching cv setting.")
                    images.append(data["images"])
        self.sensors = np.concatenate(sensors, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.images  = np.concatenate(images,  axis=0) if has_images else None
        logger.info("Dataset loaded: {} frames, image shape {}, sensors {}, actions {}",
                    len(self.actions),
                    None if self.images is None else self.images.shape[1:],
                    self.sensors.shape[1:],
                    self.actions.shape[1:])

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "sensors": self.sensors[idx],
            "actions": self.actions[idx],
        }
        if self.images is not None:
            item["image"] = self.images[idx]
        return item


def _collate(batch: list[dict]) -> dict:
    """Stack a list of per-frame dicts into torch tensors. Keeps the obs as a
    nested dict so SB3's MultiInputPolicy can consume it directly."""
    out: dict = {"actions": torch.from_numpy(np.stack([b["actions"] for b in batch])).long()}
    sensors = torch.from_numpy(np.stack([b["sensors"] for b in batch])).float()
    if "image" in batch[0]:
        images = torch.from_numpy(np.stack([b["image"] for b in batch]))
        out["obs"] = {"image": images, "sensors": sensors}
    else:
        out["obs"] = sensors
    return out


# ── Forward-pass helper that handles SB3's shared vs split feature extractor ─
def _policy_forward(policy, obs):
    """Run obs through the SB3 ActorCriticPolicy backbone and return the
    MultiCategoricalDistribution for the action head. Mirrors the start of
    `policy.forward()` so we get the same numerical path PPO uses at rollout."""
    features = policy.extract_features(obs)
    if getattr(policy, "share_features_extractor", True):
        latent_pi, _ = policy.mlp_extractor(features)
    else:
        # features is a (pi_features, vf_features) tuple in this configuration.
        pi_features, _ = features
        latent_pi = policy.mlp_extractor.forward_actor(pi_features)
    return policy._get_action_dist_from_latent(latent_pi)


# ── Training loop ────────────────────────────────────────────────────────────
def train_bc(args: argparse.Namespace) -> None:
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: {}", device)

    # 1. Discover chunks
    data_dir = Path(args.data_dir)
    chunk_files = sorted(data_dir.glob("chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.npz files in {data_dir}")

    # Detect whether the recording included images. cv_enabled in BC must
    # match the env's cv_enabled, else policy shapes won't line up.
    with np.load(chunk_files[0]) as first:
        has_images = "images" in first.files
    if has_images != cfg.cv_enabled:
        logger.warning(
            "Dataset cv mismatch: chunks have images={}, Config.cv_enabled={}. "
            "Falling back to dataset's setting for this BC run.",
            has_images, cfg.cv_enabled,
        )
    cv_enabled = has_images

    # 2. Dataset + DataLoader
    full = BCDataset(chunk_files, has_images=cv_enabled)
    n_total = len(full)
    n_val   = int(n_total * args.val_split)
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(0)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=gen)
    logger.info("Split: train={}  val={}", n_train, n_val)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=_collate, num_workers=args.num_workers, drop_last=False) \
        if n_val > 0 else None

    # 3. Stub env + PPO with same hyperparams as train.py so the resulting
    # policy state_dict drops straight into the real training run.
    env = BCStubEnv(cv_enabled, cfg.image_size, cfg.frame_stack)
    if cv_enabled:
        policy_name = "MultiInputPolicy"
        policy_kwargs = dict(
            features_extractor_kwargs=dict(cnn_output_dim=256),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
        )
    else:
        policy_name = "MlpPolicy"
        policy_kwargs = None

    model = PPO(
        policy_name, env,
        verbose=0,
        learning_rate=args.lr,
        n_steps=64, batch_size=32,  # placeholder; we never call learn()
        device=device,
        policy_kwargs=policy_kwargs,
    )
    policy = model.policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # 4. Training loop
    def _run_epoch(loader, train: bool) -> tuple[float, list[float]]:
        policy.train(train)
        total_loss = 0.0
        per_head_correct = [0] * 5
        per_head_total   = [0] * 5
        for batch in loader:
            actions = batch["actions"].to(device)
            if isinstance(batch["obs"], dict):
                obs = {k: v.to(device) for k, v in batch["obs"].items()}
            else:
                obs = batch["obs"].to(device)

            with torch.set_grad_enabled(train):
                dist = _policy_forward(policy, obs)
                # MultiCategorical: .distribution is a list of Categoricals.
                heads = dist.distribution
                losses = []
                for i, cat in enumerate(heads):
                    losses.append(F.cross_entropy(cat.logits, actions[:, i]))
                    preds = cat.logits.argmax(-1)
                    per_head_correct[i] += (preds == actions[:, i]).sum().item()
                    per_head_total[i]   += actions.size(0)
                loss = sum(losses)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * actions.size(0)

        avg_loss = total_loss / max(per_head_total[0], 1)
        accs = [c / t for c, t in zip(per_head_correct, per_head_total)]
        return avg_loss, accs

    HEAD_NAMES = ["move", "jump", "jetpack", "kick", "wand"]
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_accs = _run_epoch(train_loader, train=True)
        if val_loader is not None:
            va_loss, va_accs = _run_epoch(val_loader, train=False)
            logger.info(
                "Epoch {:3d}  train loss={:.4f}  val loss={:.4f}  "
                "val acc=[{}]",
                epoch, tr_loss, va_loss,
                " ".join(f"{n}:{a:.2f}" for n, a in zip(HEAD_NAMES, va_accs)),
            )
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                _save(policy, args.output)
                logger.info("  ↳ new best val loss, saved → {}", args.output)
        else:
            logger.info(
                "Epoch {:3d}  train loss={:.4f}  train acc=[{}]",
                epoch, tr_loss,
                " ".join(f"{n}:{a:.2f}" for n, a in zip(HEAD_NAMES, tr_accs)),
            )
            _save(policy, args.output)

    logger.info("BC training complete. Final weights at {}", args.output)
    logger.info("Warm-start RL with:  python train.py --fresh --bc-weights {}",
                args.output)


def _save(policy, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Behavioural cloning trainer for NoitaRL.")
    p.add_argument("--data-dir", type=str, default="data/bc_dataset",
                   help="Directory containing chunk_*.npz files from record_human.py.")
    p.add_argument("--output", type=str, default="checkpoints/bc_pretrained.pth",
                   help="Where to save the BC-trained policy state_dict.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of demo held out for validation (0 = no holdout).")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. Default 0 (Windows-safe).")
    return p.parse_args()


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    train_bc(parse_args())


if __name__ == "__main__":
    main()
