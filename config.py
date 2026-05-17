
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Config(BaseSettings):
    """
    All runtime settings.  Values are loaded (in priority order) from:
      1. Environment variables
      2. .env file in the working directory
      3. Defaults defined here
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Noita connection ──────────────────────────────────────────────────────
    noita_host: str       = "localhost"
    noita_base_port: int  = Field(5001, ge=1024, le=65535)
    n_envs: int           = Field(1, ge=1, le=16)

    # ── Training ──────────────────────────────────────────────────────────────
    total_timesteps: int  = Field(1_000_000, ge=1_000)
    run_name: Optional[str] = None           # human-readable experiment name
    resume_from: Optional[str] = None        # path to .zip checkpoint

    # PPO hyperparameters
    learning_rate: float = Field(1e-4, gt=0)
    n_steps: int         = Field(1024, ge=64)    # more on-policy data per update
    batch_size: int      = Field(256, ge=16)     # better GPU utilisation
    n_epochs: int        = Field(10, ge=1)
    gamma: float         = Field(0.99, ge=0, le=1)
    gae_lambda: float    = Field(0.95, ge=0, le=1)
    clip_range: float    = Field(0.2, gt=0, lt=1)
    ent_coef: float      = Field(0.03, ge=0.0, le=1.0)   # raised: 0.015 too low after policy converges
    vf_coef: float       = Field(0.5,  gt=0.0)
    max_grad_norm: float = Field(0.5,  gt=0.0)

    # ── Computer Vision (image obs branch) ───────────────────────────────────
    cv_enabled: bool      = True
    image_size: int       = Field(84, ge=32, le=256)    # square HxW for the CNN input
    image_channels: int   = Field(1, ge=1, le=3)         # 1=grayscale, 3=RGB
    frame_stack: int      = Field(4, ge=1, le=8)         # temporal stack fed to NatureCNN

    # ── Checkpoints ───────────────────────────────────────────────────────────
    checkpoint_dir: str  = "./checkpoints"
    checkpoint_freq: int = Field(100_000, ge=1_000)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_dir: str         = "./logs"
    tensorboard_dir: str = "./noita_ppo_tensorboard"
    log_level: str       = "INFO"

    # ── Telegram ──────────────────────────────────────────────────────────────
    telegram_token: str    = ""
    telegram_chat_id: str  = ""
    telegram_notify_every: int = Field(100_000, ge=1_000)

    # ── Groq (AI captions for video highlights) ───────────────────────────────
    groq_api_key: str = ""   # set GROQ_API_KEY env var or in .env file

    # ── Azure Telemetry ───────────────────────────────────────────────────────
    # Leave blank to disable telemetry (silent no-op).
    azure_cosmos_url: str            = ""
    azure_cosmos_key: str            = ""
    azure_cosmos_db: str             = "noitarl"
    azure_cosmos_container: str      = "episodes"
    azure_blob_connection_string: str = ""
    azure_blob_container_steps: str  = "noita-steps"
    azure_blob_container_assets: str = "noita-assets"

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat_id)

    # ── Weights & Biases ──────────────────────────────────────────────────────
    wandb_enabled: bool          = False
    wandb_project: str           = "noitarl"
    wandb_entity: Optional[str]  = None

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def ports(self) -> list[int]:
        return [self.noita_base_port + i for i in range(self.n_envs)]

    def effective_run_name(self) -> str:
        import time
        base = self.run_name or f"ppo_{self.n_envs}env"
        return f"{base}_{int(time.time())}"
