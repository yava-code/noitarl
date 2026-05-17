"""
SB3 callback that uploads model checkpoints to Azure Blob Storage.
Attaches to CheckpointCallback's save_freq rhythm by watching for new .zip files.
"""

from __future__ import annotations

import glob
import os
import time
from typing import Optional

from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback


class AzureCheckpointCallback(BaseCallback):
    """
    After every rollout, checks whether a new checkpoint .zip appeared in
    checkpoint_dir and queues it for upload to Azure Blob Storage.

    Designed to run alongside SB3's CheckpointCallback in a CallbackList.
    Does not duplicate CheckpointCallback logic — it just watches the directory.
    """

    def __init__(
        self,
        telemetry,
        checkpoint_dir: str = "./checkpoints",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._tel            = telemetry
        self._checkpoint_dir = checkpoint_dir
        self._uploaded: set[str] = set()

    def _on_step(self) -> bool:
        return True   # per-step hook is a no-op; work happens in _on_rollout_end

    def _on_rollout_end(self) -> None:
        self._upload_new_checkpoints()

    def _on_training_end(self) -> None:
        self._upload_new_checkpoints()

    def _upload_new_checkpoints(self) -> None:
        zips = glob.glob(os.path.join(self._checkpoint_dir, "*.zip"))
        for path in zips:
            if path in self._uploaded:
                continue
            blob_name = f"checkpoints/{os.path.basename(path)}"
            self._tel.upload_file_as_asset(blob_name, path)
            self._uploaded.add(path)
            logger.debug("[azure] Queued checkpoint upload → {}", blob_name)
