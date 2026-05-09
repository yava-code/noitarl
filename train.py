from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from noita_env import NoitaEnv
import os

def train():
    # 1. Initialize environment
    print("Initializing NoitaEnv...")
    env = NoitaEnv()
    
    # 2. Basic check (optional but recommended)
    # check_env(env) 
    
    # 3. Create model
    # MlpPolicy is suitable for vector observations
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./noita_ppo_tensorboard/"
    )
    
    # 4. Train
    print("Starting training loop (10,000 steps)...")
    try:
        model.learn(total_timesteps=10000, progress_bar=True)
        print("Training complete!")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    # 5. Save model
    model_path = "noita_ppo_mvp"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

if __name__ == "__main__":
    train()
