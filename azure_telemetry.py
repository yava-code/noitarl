"""
Asynchronous Azure telemetry pipeline for NoitaRL.

Zero-blocking design: all I/O runs in a background worker thread.
The training loop calls log_step() and flush_episode() which only do
queue.put() — they never block on network I/O.

Azure services used:
  - Cosmos DB (NoSQL, serverless): one document per episode summary
  - Blob Storage: per-episode compressed JSONL (steps) + assets (GIFs, checkpoints)

When Azure credentials are absent the class operates as a silent no-op.
"""

from __future__ import annotations

import gzip
import io
import json
import queue
import threading
import time
import uuid
from typing import Any, Optional
from loguru import logger


_SENTINEL = object()   # signals worker to shut down


class AzureTelemetry:
    """
    Thread-safe, zero-blocking telemetry uploader.

    Usage:
        tel = AzureTelemetry(config)
        tel.log_step(step_dict)                 # non-blocking
        tel.flush_episode(episode_info_dict)    # non-blocking
        tel.upload_asset("my.gif", gif_bytes)   # non-blocking
        tel.shutdown()                          # call once at training end
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._enabled = self._check_credentials(cfg)
        self._session_id = str(uuid.uuid4())
        self._step_buf: list[dict] = []       # per-episode in-memory buffer
        self._buf_lock = threading.Lock()
        self._work_q: queue.Queue = queue.Queue()

        if self._enabled:
            self._cosmos  = self._make_cosmos_client(cfg)
            self._blob    = self._make_blob_client(cfg)
            self._worker  = threading.Thread(
                target=self._run, daemon=True, name="azure-tel"
            )
            self._worker.start()
            logger.info(
                "[azure] Telemetry enabled. session_id={} cosmos={} blob={}",
                self._session_id,
                cfg.azure_cosmos_db,
                cfg.azure_blob_container_steps,
            )
        else:
            self._cosmos = None
            self._blob   = None
            self._worker = None
            logger.debug("[azure] Credentials absent — telemetry is a no-op.")

    # ── Public API (all non-blocking) ─────────────────────────────────────────

    def log_step(self, step_data: dict) -> None:
        """Buffer one training step. O(1), never touches the network."""
        if not self._enabled:
            return
        record = {
            "session_id": self._session_id,
            "ts": time.time(),
            **step_data,
        }
        with self._buf_lock:
            self._step_buf.append(record)

    def flush_episode(self, episode_info: dict) -> None:
        """
        Queue upload of the current episode's step buffer + episode summary.
        Returns immediately; upload happens in background worker.
        """
        if not self._enabled:
            return
        with self._buf_lock:
            steps = self._step_buf.copy()
            self._step_buf.clear()

        episode_num = episode_info.get("noita/episode", 0)
        blob_name   = f"{self._session_id}/ep{episode_num:06d}.jsonl.gz"

        self._work_q.put({
            "type":        "episode",
            "steps":       steps,
            "blob_name":   blob_name,
            "summary":     self._make_summary(episode_info),
        })

    def upload_asset(
        self,
        blob_name: str,
        data: bytes,
        container: Optional[str] = None,
    ) -> None:
        """
        Queue upload of arbitrary bytes (GIF, checkpoint) to the assets container.
        Returns immediately.
        """
        if not self._enabled:
            return
        self._work_q.put({
            "type":      "asset",
            "blob_name": blob_name,
            "data":      data,
            "container": container or self._cfg.azure_blob_container_assets,
        })

    def upload_file_as_asset(self, blob_name: str, local_path: str) -> None:
        """Read a local file and queue it for Blob upload. Returns immediately."""
        if not self._enabled:
            return
        try:
            with open(local_path, "rb") as f:
                data = f.read()
            self.upload_asset(blob_name, data)
        except OSError as exc:
            logger.warning("[azure] Could not read {}: {}", local_path, exc)

    def shutdown(self, timeout: float = 60.0) -> None:
        """Block until all queued work is uploaded, then stop the worker."""
        if not self._enabled or self._worker is None:
            return
        logger.info("[azure] Flushing telemetry queue before shutdown…")
        self._work_q.put(_SENTINEL)
        self._worker.join(timeout=timeout)
        if self._worker.is_alive():
            logger.warning("[azure] Worker did not finish within {}s", timeout)
        logger.info("[azure] Telemetry worker shut down.")

    # ── Background worker ─────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            try:
                item = self._work_q.get(timeout=30)
            except queue.Empty:
                continue

            if item is _SENTINEL:
                break

            try:
                if item["type"] == "episode":
                    self._upload_episode(item)
                elif item["type"] == "asset":
                    self._upload_asset_item(item)
            except Exception as exc:
                logger.warning("[azure] Worker error (item skipped): {}", exc)
            finally:
                self._work_q.task_done()

    def _upload_episode(self, item: dict) -> None:
        steps    = item["steps"]
        summary  = item["summary"]
        blob_name = item["blob_name"]

        # 1. Compress step JSONL → Blob Storage
        if steps and self._blob is not None:
            try:
                buf = io.BytesIO()
                with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
                    for rec in steps:
                        gz.write((json.dumps(rec) + "\n").encode())
                buf.seek(0)
                container_client = self._blob.get_container_client(
                    self._cfg.azure_blob_container_steps
                )
                self._ensure_container(container_client)
                container_client.upload_blob(
                    name=blob_name,
                    data=buf.getvalue(),
                    overwrite=True,
                    content_settings=self._gzip_content_settings(),
                )
                logger.debug(
                    "[azure] Uploaded {} steps → blob://{}/{}",
                    len(steps), self._cfg.azure_blob_container_steps, blob_name,
                )
            except Exception as exc:
                logger.warning("[azure] Blob episode upload failed: {}", exc)

        # 2. Upsert episode summary → Cosmos DB
        if self._cosmos is not None:
            try:
                db  = self._cosmos.get_database_client(self._cfg.azure_cosmos_db)
                ctr = db.get_container_client(self._cfg.azure_cosmos_container)
                ctr.upsert_item(summary)
                logger.debug(
                    "[azure] Cosmos upsert episode {} session {}",
                    summary.get("episode"), self._session_id[:8],
                )
            except Exception as exc:
                logger.warning("[azure] Cosmos upsert failed: {}", exc)

    def _upload_asset_item(self, item: dict) -> None:
        if self._blob is None:
            return
        try:
            container_name = item["container"]
            container_client = self._blob.get_container_client(container_name)
            self._ensure_container(container_client)
            container_client.upload_blob(
                name=item["blob_name"],
                data=item["data"],
                overwrite=True,
            )
            logger.debug(
                "[azure] Uploaded asset → blob://{}/{}",
                container_name, item["blob_name"],
            )
        except Exception as exc:
            logger.warning("[azure] Asset upload failed: {}", exc)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _check_credentials(cfg) -> bool:
        return bool(
            getattr(cfg, "azure_cosmos_url", None) or
            getattr(cfg, "azure_blob_connection_string", None)
        )

    @staticmethod
    def _make_cosmos_client(cfg):
        try:
            from azure.cosmos import CosmosClient
            url = getattr(cfg, "azure_cosmos_url", "")
            key = getattr(cfg, "azure_cosmos_key", "")
            if url and key:
                return CosmosClient(url, credential=key)
        except ImportError:
            logger.warning("[azure] azure-cosmos not installed — Cosmos uploads disabled.")
        return None

    @staticmethod
    def _make_blob_client(cfg):
        try:
            from azure.storage.blob import BlobServiceClient
            conn = getattr(cfg, "azure_blob_connection_string", "")
            if conn:
                return BlobServiceClient.from_connection_string(conn)
        except ImportError:
            logger.warning("[azure] azure-storage-blob not installed — Blob uploads disabled.")
        return None

    @staticmethod
    def _ensure_container(container_client) -> None:
        try:
            container_client.create_container()
        except Exception:
            pass   # already exists

    @staticmethod
    def _gzip_content_settings():
        try:
            from azure.storage.blob import ContentSettings
            return ContentSettings(content_encoding="gzip", content_type="application/jsonlines")
        except ImportError:
            return None

    def _make_summary(self, info: dict) -> dict:
        ep = info.get("episode", {})
        return {
            "id":              f"{self._session_id}_ep{info.get('noita/episode', 0):06d}",
            "session_id":      self._session_id,
            "ts":              time.time(),
            "episode":         info.get("noita/episode", 0),
            "global_step":     info.get("noita/global_step", 0),
            "reward":          float(ep.get("r", 0.0)),
            "length":          int(ep.get("l", 0)),
            "visited_chunks":  int(info.get("noita/visited_chunks", 0)),
            "max_spawn_distance": float(info.get("noita/max_spawn_distance", 0.0)),
            "max_depth":       float(info.get("noita/max_depth", 0.0)),
            "kills":           int(info.get("noita/kills", 0)),
            "chests_opened":   int(info.get("noita/chests_opened", 0)),
            "total_damage":    float(info.get("noita/total_damage", 0.0)),
            "run_time_s":      float(info.get("noita/run_time_s", 0.0)),
            "death_reason":    info.get("noita/death_reason", "UNKNOWN"),
            "reward_breakdown": info.get("noita/reward_breakdown", {}),
            "route_x":         info.get("noita/route_x", []),
            "route_y":         info.get("noita/route_y", []),
        }
