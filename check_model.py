from stable_baselines3 import PPO
import numpy as np

model = PPO.load('best_model.zip')

print("=" * 60)
print("Model Configuration")
print("=" * 60)
print(f"Observation space: {model.observation_space}")
print(f"Action space: {model.action_space}")
print(f"Policy kwargs: {model.policy_kwargs}")
print(f"n_steps: {model.n_steps}")
print(f"batch_size: {model.batch_size}")
print(f"n_epochs: {model.n_epochs}")
print(f"gamma: {model.gamma}")
print(f"gae_lambda: {model.gae_lambda}")
print(f"clip_range: {model.clip_range}")
print(f"ent_coef: {model.ent_coef}")
print(f"learning_rate: {model.lr_schedule}")
print(f"device: {model.device}")
print("=" * 60)
