"""Phase 0 smoke test: PPO on CartPole-v1 should reach a mean reward of 500.

Hyperparameters from the RL Baselines3 Zoo. Takes a few minutes on CPU.
"""
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

SEED = 0

env = make_vec_env("CartPole-v1", n_envs=8, seed=SEED)
model = PPO(
    "MlpPolicy", env,
    n_steps=32, batch_size=256, n_epochs=20,
    learning_rate=1e-3, gamma=0.98, gae_lambda=0.8,
    clip_range=0.2, ent_coef=0.0,
    tensorboard_log="runs/", seed=SEED, verbose=0,
)
model.learn(total_timesteps=100_000, tb_log_name="cartpole_smoke")

mean, std = evaluate_policy(model, make_vec_env("CartPole-v1", n_envs=1, seed=SEED + 1),
                            n_eval_episodes=10, deterministic=True)
print(f"CartPole-v1 mean reward: {mean:.1f} +- {std:.1f}  ->  {'PASS' if mean >= 500 else 'FAIL'}")
