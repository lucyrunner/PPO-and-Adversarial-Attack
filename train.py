from __future__ import annotations
import argparse, os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from acc_env import ACCEnv

def make_env(brake_profile=False, normalize_obs=True):
    def _thunk():
        return ACCEnv(brake_profile=brake_profile, normalize_obs=normalize_obs)
    return _thunk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-steps', type=int, default=300_000)
    parser.add_argument('--logdir', type=str, default='runs/ppo_baseline')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    # Vectorized env (single for simplicity)
    env = DummyVecEnv([make_env(brake_profile=False, normalize_obs=True)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1.0)

    model = PPO(
        "MlpPolicy", env, verbose=1, seed=args.seed,
        n_steps=1024, batch_size=128, learning_rate=3e-4, gamma=0.99,
        gae_lambda=0.95, clip_range=0.2, ent_coef=0.0
    )
    new_logger = configure(args.logdir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=args.total_steps, progress_bar=True)

    # Save both model and normalization stats
    model.save(os.path.join(args.logdir, 'ppo_acc'))
    env.save(os.path.join(args.logdir, 'vecnormalize.pkl'))
    print(f"Saved to {args.logdir}")

if __name__ == '__main__':
    main()
