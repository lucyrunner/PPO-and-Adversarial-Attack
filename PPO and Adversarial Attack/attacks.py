from __future__ import annotations
from __future__ import annotations
#!/usr/bin/env python
# coding: utf-8

# In[82]:


# One-cell train + save + smoke-check for ACCEnv + PPO (robust reset/step handling)

import os, sys, subprocess, shlex
from IPython import get_ipython

# 1) Ensure ACCEnv is importable
NOTEBOOK = "acc_env.ipynb"
try:
    from acc_env import ACCEnv  # prefer acc_env.py if present
    print("Imported ACCEnv from acc_env.py")
except Exception:
    if os.path.exists(NOTEBOOK):
        print("Converting acc_env.ipynb -> acc_env.py ...")
        subprocess.run(shlex.split(f"jupyter nbconvert --to python {NOTEBOOK}"), check=True)
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())
        from acc_env import ACCEnv
        print("Imported ACCEnv from converted acc_env.py")
    else:
        print("No acc_env.py found; running acc_env.ipynb ...")
        get_ipython().run_line_magic("run", f"./{NOTEBOOK}")
        if "ACCEnv" in globals():
            print("ACCEnv defined by running acc_env.ipynb")
        else:
            raise FileNotFoundError("ACCEnv not found. Put acc_env.py or acc_env.ipynb in this folder.")

# 2) Imports for training
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure

# 3) Hyperparameters & paths
LOGDIR = "runs/ppo_baseline"
os.makedirs(LOGDIR, exist_ok=True)
TOTAL_TIMESTEPS = 200_000
SEED = 42
N_ENVS = 1

PPO_PARAMS = dict(
    policy="MlpPolicy",
    verbose=1,
    seed=SEED,
    n_steps=1024,
    batch_size=128,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
)

# 4) Env factory
# NOTE:
# - Do NOT pass brake_profile (ACCEnv has no such arg).
# - Keep normalize_obs=False; let VecNormalize handle normalization.
def make_env(seed=0, normalize_obs=False):
    def _thunk():
        env = ACCEnv(normalize_obs=normalize_obs, seed=seed)
        return env
    return _thunk

# 5) Create vectorized normalized training env
# Use distinct seeds if N_ENVS > 1 to avoid identical instances
base_env = DummyVecEnv([make_env(seed=SEED + i, normalize_obs=False) for i in range(N_ENVS)])
# Use a sane clip range; 10.0 is SB3 default. 1.0 over-clips and can hide attack effects.
train_env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
print("Train env created. obs_space:", train_env.observation_space, "act_space:", train_env.action_space)

# 6) Create model, set logger, train
model = PPO(**PPO_PARAMS, env=train_env)
model.set_logger(configure(LOGDIR, ["stdout", "csv", "tensorboard"]))
print(f"Starting training for {TOTAL_TIMESTEPS} timesteps ...")
model.learn(total_timesteps=TOTAL_TIMESTEPS)
print("Training finished.")

# 7) Save model + VecNormalize
model_path = os.path.join(LOGDIR, "ppo_acc")
vec_path = os.path.join(LOGDIR, "vecnormalize.pkl")
model.save(model_path)        # writes ppo_acc.zip
train_env.save(vec_path)      # writes vecnormalize.pkl
print("Saved model ->", model_path + ".zip")
print("Saved VecNormalize ->", vec_path)

# --------------------- Smoke-check (fixed) ---------------------

# Robust wrappers for reset/step across gym/sb3 versions
def reset_unwrap(env, **kwargs):
    out = env.reset(**kwargs)
    if isinstance(out, tuple) and len(out) == 2:
        obs, _info = out
        return obs
    return out  # obs only

def step_unwrap(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated), bool(truncated), info
    elif len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, bool(done), False, info
    else:
        raise RuntimeError(f"Unexpected env.step() return length: {len(out)}")

# Build an evaluation env that uses the saved normalization stats
eval_base = DummyVecEnv([make_env(seed=123, normalize_obs=False)])
eval_env = VecNormalize.load(vec_path, eval_base)
eval_env.training = False
eval_env.norm_reward = False

# load saved model, bound to eval_env
model_loaded = PPO.load(model_path + ".zip", env=eval_env)
print("Loaded model and VecNormalize for evaluation.")

def run_eval_episode(mdl, env, max_steps=1000):
    obs = reset_unwrap(env)
    total_r = 0.0
    collided = False
    steps = 0
    while True:
        action, _ = mdl.predict(obs, deterministic=True)
        obs, r, term, trunc, info = step_unwrap(env, action)
        total_r += float(r[0]) if hasattr(r, "__len__") else float(r)
        steps += 1
        # info can be list[dict] for VecEnv
        idict = info if isinstance(info, dict) else (info[0] if hasattr(info, "__len__") and len(info) else {})
        if idict.get("collision", False):
            collided = True
        if term or trunc or (steps >= max_steps):
            break
    return total_r, collided, steps

print("Running 3 smoke-check episodes ...")
for i in range(3):
    ret, coll, steps = run_eval_episode(model_loaded, eval_env)
    print(f"Episode {i}: return={ret:.3f}, collision={coll}, steps={steps}")

# Optional: quick plot of training CSV if available
csv_path = os.path.join(LOGDIR, "progress.csv")
if os.path.exists(csv_path):
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if 'rollout/ep_rew_mean' in df.columns:
            plt.figure(figsize=(8,3))
            plt.plot(df['rollout/ep_rew_mean'])
            plt.title("rollout/ep_rew_mean during training")
            plt.xlabel("logging step")
            plt.ylabel("mean episode reward")
            plt.show()
    except Exception:
        pass

print("\nDone. You now have:")
print(" -", model_path + ".zip")
print(" -", vec_path)
print("These artifacts will be loaded by your attacks notebook.")


# In[83]:


# === Make model/env ready for the demo ===
import os, sys, subprocess, shlex
from IPython import get_ipython

# 1) Ensure ACCEnv is defined (import .py; else convert .ipynb -> .py; else %run notebook)
try:
    from acc_env import ACCEnv  # if you already have acc_env.py alongside this notebook
    print("Imported ACCEnv from acc_env.py")
except ModuleNotFoundError:
    if os.path.exists("acc_env.ipynb"):
        print("Converting acc_env.ipynb -> acc_env.py ...")
        subprocess.run(shlex.split("jupyter nbconvert --to python acc_env.ipynb"), check=True)
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())
        from acc_env import ACCEnv
        print("Imported ACCEnv from converted acc_env.py")
    else:
        print("Running acc_env.ipynb directly...")
        get_ipython().run_line_magic("run", "./acc_env.ipynb")
        # ACCEnv should now be in globals


# In[84]:


# 2) Try to load saved PPO + VecNormalize; otherwise quick-train a small model
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

LOGDIR = "runs/ppo_baseline"          # change if you saved elsewhere
vec_path = os.path.join(LOGDIR, "vecnormalize.pkl")
mdl_path = os.path.join(LOGDIR, "ppo_acc.zip")

def make_env(seed=123, with_braking=True, normalize_obs=False):
    """
    Factory returning a fresh ACCEnv instance.
    - normalize_obs=False so VecNormalize handles scaling (recommended with SB3).
    - If with_braking=True, attach a simple lead-vehicle braking profile via attribute.
      NOTE: Ensure ACCEnv.step() uses:
            lead_acc = float(self.lead_profile(self.current_step)) if hasattr(self, "lead_profile") else 0.0
            self.v_lead = float(np.clip(self.v_lead + lead_acc * DT, 0.0, 100.0))
            self.x_lead = float(self.x_lead + self.v_lead * DT + 0.5 * lead_acc * DT * DT)
    """
    def _thunk():
        env = ACCEnv(normalize_obs=normalize_obs, seed=seed)
        if with_braking:
            def lead_profile(step):
                # mild braking between steps ~120–180; tweak as needed
                return -2.5 if 120 <= step <= 180 else 0.0
            env.lead_profile = lead_profile
        return env
    return _thunk

model = None
env = None

if os.path.exists(vec_path) and os.path.exists(mdl_path):
    print(f"Loading saved model/env from {LOGDIR} ...")
    # Build a base env (raw obs) and load VecNormalize stats onto it
    base_env = DummyVecEnv([make_env(seed=123, with_braking=True, normalize_obs=False)])
    env = VecNormalize.load(vec_path, base_env)
    env.training = False
    env.norm_reward = False

    # Load PPO with the normalized eval env
    model = PPO.load(mdl_path, env=env)
    print("Loaded saved PPO and VecNormalize.")
else:
    print("Saved files not found; quick-training a small model so the demo can run...")

    # ---- Train (no braking) ----
    train_base = DummyVecEnv([make_env(seed=42, with_braking=False, normalize_obs=False)])
    # Let VecNormalize handle observation scaling; use a generous clip (default 10.0)
    train_env = VecNormalize(train_base, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", train_env, seed=42, verbose=0,
        n_steps=512, batch_size=128, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0
    )
    model.learn(total_timesteps=8_000)  # quick smoke-train

    # ---- Eval env (with braking) sharing the same normalization stats ----
    eval_base = DummyVecEnv([make_env(seed=123, with_braking=True, normalize_obs=False)])
    env = VecNormalize(eval_base, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # Copy normalization statistics from training env so observations match what the policy expects
    env.obs_rms = train_env.obs_rms
    env.ret_rms = train_env.ret_rms
    env.training = False

    print("Quick train done; model/env are ready.")

print("\n✅ model and env are ready in this kernel.")



# In[85]:


import gymnasium as gym
import torch
import numpy as np
from typing import Any


# In[86]:


#imports & setup for attacks
import numpy as np
import torch

# Choose device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _to_obs_tensor(obs: np.ndarray) -> torch.Tensor:
    """
    Convert a single normalized observation (shape: (obs_dim,) or (1, obs_dim))
    into a torch tensor with requires_grad=True, on DEVICE, batched as (1, obs_dim).
    """
    if isinstance(obs, np.ndarray):
        t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
    else:
        t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    t.requires_grad_(True)
    return t

def _clip_norm_obs(obs_adv: np.ndarray, low=-1.0, high=1.0) -> np.ndarray:
    """Clip perturbed normalized obs back into valid range (VecNormalize space)."""
    return np.clip(obs_adv, low, high, dtype=np.float32)

def _sign_np(x: np.ndarray) -> np.ndarray:
    """Stable sign for numpy arrays."""
    s = np.zeros_like(x, dtype=np.float32)
    s[x > 0] = 1.0
    s[x < 0] = -1.0
    return s


# In[87]:


def fgsm_attack(model, obs_norm: np.ndarray, epsilon: float = 0.01, clip_to_unit=True) -> np.ndarray:
    """
    FGSM: s' = s + ε * sign(∇_s μθ(s))
    - model: SB3 PPO model (with .policy)
    - obs_norm: SINGLE normalized observation that the policy consumes (VecNormalize output)
    - epsilon: perturbation budget in normalized units
    """
    policy = model.policy
    policy.eval()
    policy.to(DEVICE)

    obs_t = _to_obs_tensor(obs_norm)  # (1, obs_dim), requires_grad=True

    # Get policy distribution and its mean action
    dist = policy.get_distribution(obs_t)
    # Many SB3 policies expose .distribution.mean for Gaussian policies
    if hasattr(dist.distribution, "mean"):
        act_mean = dist.distribution.mean   # shape (1, act_dim)
        # To backprop to obs, reduce to scalar; sum across action dims
        scalar = act_mean.sum()
    else:
        # Fallback: use the deterministic action as proxy (less ideal)
        # This path should rarely be needed for standard MlpPolicy (Gaussian)
        act = policy.predict(obs_t, deterministic=True)[0]
        scalar = act.sum()

    # Backprop
    policy.zero_grad(set_to_none=True)
    if obs_t.grad is not None:
        obs_t.grad.zero_()
    scalar.backward(retain_graph=False)

    grad = obs_t.grad.detach().cpu().numpy().squeeze(0)  # (obs_dim,)
    pert = epsilon * _sign_np(grad)
    adv = (obs_norm + pert).astype(np.float32)

    if clip_to_unit:
        adv = _clip_norm_obs(adv, -1.0, 1.0)
    return adv


def oia_attack(model, obs_norm: np.ndarray, epsilon: float = 0.01, clip_to_unit=True) -> np.ndarray:
    """
    OIA: s' = s + ε * sign(∇_s Vϕ(s))
    - model: SB3 PPO model (with .policy)
    - obs_norm: SINGLE normalized observation (VecNormalize output)
    - epsilon: perturbation budget in normalized units
    """
    policy = model.policy
    policy.eval()
    policy.to(DEVICE)

    obs_t = _to_obs_tensor(obs_norm)  # (1, obs_dim), requires_grad=True

    # Critic value forward pass
    # SB3 PPO policies usually have .predict_values(obs) -> (batch, 1) tensor
    if hasattr(policy, "predict_values"):
        v = policy.predict_values(obs_t)
    elif hasattr(policy, "forward_critic"):
        v = policy.forward_critic(obs_t)
    else:
        # Fallback: try value_net directly (MlpPolicy exposes critic internally)
        v = policy.value_net(policy.extract_features(obs_t))
    v_scalar = v.sum()

    # Backprop wrt obs
    policy.zero_grad(set_to_none=True)
    if obs_t.grad is not None:
        obs_t.grad.zero_()
    v_scalar.backward(retain_graph=False)

    grad = obs_t.grad.detach().cpu().numpy().squeeze(0)  # (obs_dim,)
    pert = epsilon * _sign_np(grad)
    adv = (obs_norm + pert).astype(np.float32)

    if clip_to_unit:
        adv = _clip_norm_obs(adv, -1.0, 1.0)
    return adv


# In[88]:


# Thin wrappers to match your eval loop signature attack_fn(obs, eps)

class FGSMAttack:
    """Callable wrapper: adv_obs = FGSMAttack(model)(obs_norm, eps)"""
    def __init__(self, model, clip_to_unit=True):
        self.model = model
        self.clip = clip_to_unit

    def __call__(self, obs_norm: np.ndarray, eps: float = 0.01) -> np.ndarray:
        return fgsm_attack(self.model, obs_norm, epsilon=eps, clip_to_unit=self.clip)


class OIAttack:
    """Callable wrapper: adv_obs = OIAttack(model)(obs_norm, eps)"""
    def __init__(self, model, clip_to_unit=True):
        self.model = model
        self.clip = clip_to_unit

    def __call__(self, obs_norm: np.ndarray, eps: float = 0.01) -> np.ndarray:
        return oia_attack(self.model, obs_norm, epsilon=eps, clip_to_unit=self.clip)


# Convenience functions (if you prefer function refs)
def fgsm_fn(obs_norm: np.ndarray, eps: float, model=None) -> np.ndarray:
    assert model is not None, "Pass model via functools.partial or switch to FGSMAttack class."
    return fgsm_attack(model, obs_norm, epsilon=eps, clip_to_unit=True)

def oia_fn(obs_norm: np.ndarray, eps: float, model=None) -> np.ndarray:
    assert model is not None, "Pass model via functools.partial or switch to OIAttack class."
    return oia_attack(model, obs_norm, epsilon=eps, clip_to_unit=True)


# In[89]:


# Cell D: quick sanity (requires a loaded SB3 PPO model and a normalized obs)
# Example usage inside your evaluation notebook:
#   fgsm = FGSMAttack(model)
#   oia  = OIAttack(model)
#   adv_obs_fgsm = fgsm(obs_norm, eps=0.02)
#   adv_obs_oia  = oia(obs_norm,  eps=0.02)

def _debug_grad_direction(model, obs_norm):
    """Utility to verify both grads are non-zero and distinct."""
    with torch.no_grad():
        pass  # ensure no lingering graph

    # FGSM grad dir
    obs_t = _to_obs_tensor(obs_norm)
    dist = model.policy.get_distribution(obs_t)
    m = dist.distribution.mean
    model.policy.zero_grad(set_to_none=True)
    if obs_t.grad is not None: obs_t.grad.zero_()
    m.sum().backward()
    g_fgsm = obs_t.grad.detach().cpu().numpy().squeeze(0)

    # OIA grad dir
    obs_t2 = _to_obs_tensor(obs_norm)
    if hasattr(model.policy, "predict_values"):
        v = model.policy.predict_values(obs_t2)
    else:
        v = model.policy.forward_critic(obs_t2)
    model.policy.zero_grad(set_to_none=True)
    if obs_t2.grad is not None: obs_t2.grad.zero_()
    v.sum().backward()
    g_oia = obs_t2.grad.detach().cpu().numpy().squeeze(0)

    return g_fgsm, g_oia

# Example (commented):
# test_obs = np.zeros((3,), dtype=np.float32)  # replace with a real normalized obs
# gf, go = _debug_grad_direction(model, test_obs)
# print("FGSM grad L1:", np.abs(gf).sum(), "OIA grad L1:", np.abs(go).sum())

