import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from noita_env import NoitaEnv


def train():
    os.makedirs("checkpoints", exist_ok=True)

    print("[Train] Initialising NoitaEnv...")
    env = NoitaEnv()

    # Save a checkpoint every 100 000 steps so training can be resumed
    checkpoint_cb = CheckpointCallback(
        save_freq   = 100_000,
        save_path   = "./checkpoints/",
        name_prefix = "noita_ppo",
        verbose     = 1,
    )

    print("[Train] Building PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose          = 1,
        learning_rate    = 3e-4,
        n_steps          = 2048,
        batch_size       = 64,
        n_epochs         = 10,
        gamma            = 0.99,
        gae_lambda       = 0.95,
        clip_range       = 0.2,
        tensorboard_log  = "./noita_ppo_tensorboard/",
    )

    print("[Train] Starting training — 1 000 000 steps (~15 h at 18 it/s)")
    print("[Train] View live curves:  tensorboard --logdir ./noita_ppo_tensorboard/")
    try:
        model.learn(
            total_timesteps = 1_000_000,
            callback        = checkpoint_cb,
            progress_bar    = True,
        )
        print("[Train] Training complete!")
    except KeyboardInterrupt:
        print("[Train] Interrupted by user.")

    model.save("noita_ppo_final")
    print("[Train] Model saved → noita_ppo_final.zip")
    print("[Train] Resume later with:  PPO.load('noita_ppo_final').learn(...)")


if __name__ == "__main__":
    train()
