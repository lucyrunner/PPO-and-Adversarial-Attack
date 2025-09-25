from __future__ import annotations
from __future__ import annotations
#!/usr/bin/env python
# coding: utf-8

# In[55]:


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
def make_env(seed=0, brake_profile=True, normalize_obs=True):
    def _thunk():
        return ACCEnv(brake_profile=brake_profile, normalize_obs=normalize_obs, seed=seed)
    return _thunk

# 5) Create vectorized normalized training env
base_env = DummyVecEnv([make_env(seed=SEED, brake_profile=False, normalize_obs=True) for _ in range(N_ENVS)])
train_env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_obs=1.0)
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
eval_base = DummyVecEnv([make_env(seed=123, brake_profile=True, normalize_obs=True)])
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


# In[56]:


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


# In[57]:


# 2) Try to load saved PPO + VecNormalize; otherwise quick-train a small model
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

LOGDIR = "runs/ppo_baseline"  # change if you saved elsewhere
vec_path = os.path.join(LOGDIR, "vecnormalize.pkl")
mdl_path = os.path.join(LOGDIR, "ppo_acc.zip")

def make_env(seed=123, brake_profile=True, normalize_obs=True):
    def _thunk():
        return ACCEnv(brake_profile=brake_profile, normalize_obs=normalize_obs, seed=seed)
    return _thunk

model = None
env = None

if os.path.exists(vec_path) and os.path.exists(mdl_path):
    print(f"Loading saved model/env from {LOGDIR} ...")
    base_env = DummyVecEnv([make_env(seed=123, brake_profile=True, normalize_obs=True)])
    env = VecNormalize.load(vec_path, base_env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(mdl_path, env=env)
    print("Loaded saved PPO and VecNormalize.")
else:
    print("Saved files not found; doing a quick in-memory train so the demo can run...")
    # quick train on a stationary-lead scenario (no braking) so it learns *something*
    train_env = DummyVecEnv([make_env(seed=42, brake_profile=False, normalize_obs=True)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=1.0)
    model = PPO(
        "MlpPolicy", train_env, seed=42, verbose=0,
        n_steps=512, batch_size=128, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0
    )
    model.learn(total_timesteps=8_000)  # small, fast
    # build an eval env (with braking enabled) sharing the same VecNormalize statistics
    eval_base = DummyVecEnv([make_env(seed=123, brake_profile=True, normalize_obs=True)])
    env = VecNormalize(eval_base, norm_obs=True, norm_reward=False, clip_obs=1.0)
    # copy normalization stats from training env so obs scales match what policy expects
    env.obs_rms = train_env.obs_rms
    env.ret_rms = train_env.ret_rms
    env.training = False
    print("Quick train done; model/env are ready.")

print("\n✅ model and env are ready in this kernel.")


# In[58]:


import gymnasium as gym
import torch
import numpy as np
from typing import Any


# In[59]:


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


# In[60]:


class AttackWrapper:
    """Base wrapper that perturbs observations before the agent acts."""
    def __init__(self, model: Any, epsilon: float = 0.01, device: str = "cpu") -> None:
        self.model = model
        self.eps = float(epsilon)
        self.device = device

    def perturb(self, obs: np.ndarray) -> np.ndarray:
        return obs

    def act(self, obs: np.ndarray):
        # Compute adversarial observation (gradients enabled in perturb),
        # then call model.predict without gradients.
        obs_adv = self.perturb(obs)
        with torch.no_grad():
            action, _ = self.model.predict(obs_adv, deterministic=True)
        return action, obs_adv


# In[61]:


class FGSMAttack(AttackWrapper):
    """FGSM with respect to policy mean action (pre-squash)."""
    def perturb(self, obs: np.ndarray) -> np.ndarray:
        # prepare policy for gradients
        self.model.policy.set_training_mode(True)
        self.model.policy.zero_grad(set_to_none=True)

        obs_t = _to_tensor(obs)
        single = False
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
            single = True
        obs_t = obs_t.to(self.device)
        obs_t.requires_grad_(True)

        # forward through policy internals to get mean action
        features = self.model.policy.extract_features(obs_t)
        latent_pi, _ = self.model.policy.mlp_extractor(features)
        mean_actions = self.model.policy.action_net(latent_pi)  # [B, act_dim]

        # simple scalar objective: increase squared mean action
        obj = (mean_actions ** 2).sum()
        obj.backward()

        grad_sign = torch.sign(obs_t.grad)
        adv = torch.clamp(obs_t + self.eps * grad_sign, -1.0, 1.0)
        adv_np = adv.detach().cpu().numpy()
        return adv_np[0] if single else adv_np


# In[62]:


class OIAttack(AttackWrapper):
    """Optimism Induction Attack: increase the critic value V(s)."""
    def perturb(self, obs: np.ndarray) -> np.ndarray:
        self.model.policy.set_training_mode(True)
        self.model.policy.zero_grad(set_to_none=True)

        obs_t = _to_tensor(obs)
        single = False
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
            single = True
        obs_t = obs_t.to(self.device)
        obs_t.requires_grad_(True)

        features = self.model.policy.extract_features(obs_t)
        _, latent_vf = self.model.policy.mlp_extractor(features)
        values = self.model.policy.value_net(latent_vf)  # [B,1]

        obj = values.sum()
        obj.backward()

        grad_sign = torch.sign(obs_t.grad)
        adv = torch.clamp(obs_t + self.eps * grad_sign, -1.0, 1.0)
        adv_np = adv.detach().cpu().numpy()
        return adv_np[0] if single else adv_np


# In[63]:


def print_attack_sanity(model, env, eps=0.01):
    atk = FGSMAttack(model, epsilon=eps, device="cpu")
    obs = env.reset()[0]
    adv = atk.perturb(obs)
    print("FGSM sanity:")
    print(" original obs:", obs)
    print(" adv obs     :", adv)
    print(" max |Δ|     :", float(np.max(np.abs(np.array(adv) - np.array(obs)))))

    atk2 = OIAttack(model, epsilon=eps, device="cpu")
    adv2 = atk2.perturb(obs)
    print("\nOIA sanity:")
    print(" original obs:", obs)
    print(" adv obs     :", adv2)
    print(" max |Δ|     :", float(np.max(np.abs(np.array(adv2) - np.array(obs)))))


# In[64]:


print_attack_sanity(model, env, eps=0.02)


# In[65]:


# Requires: a loaded PPO `model` bound to an ACCEnv(normalize_obs=True) `env`
try:
    obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
except Exception:
    print("Load/define `model` and `env` (ACCEnv with normalize_obs=True) first.")
else:
    fgsm = FGSMAttack(model, epsilon=0.01)
    oia  = OIAttack(model,  epsilon=0.02)

    adv_f = fgsm.perturb(obs)
    adv_o = oia.perturb(obs)

    print("FGSM max |Δ|:", float(np.max(np.abs(adv_f - obs))))
    print("OIA  max |Δ|:", float(np.max(np.abs(adv_o - obs))))

    # If you want to see an actual collision tendency right here:
    env.unwrapped.set_safety_obs_for_filter(adv_o)  # make safety use attacked obs
    a_o, _ = model.predict(adv_o, deterministic=True)
    _, _, term, trunc, info = env.step(a_o)
    print("Step under OIA attacked obs — collision flag:", (info[0] if isinstance(info, list) else info).get("collision"))



# In[66]:


import torch
import numpy as np
import gymnasium as gym

class AttackWrapper(gym.Wrapper):
    def __init__(self, env, model, epsilon=0.01):
        super(AttackWrapper, self).__init__(env)
        self.model = model
        self.epsilon = epsilon
        self.attack_name = "BaseAttack"
        
    def step(self, action):
        return self.env.step(action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class FGSMAttackWrapper(AttackWrapper):
    def __init__(self, env, model, epsilon=0.01):
        super(FGSMAttackWrapper, self).__init__(env, model, epsilon)
        self.attack_name = "FGSM"
        
    def step(self, action):
        # Get current state for gradient computation
        norm_state = self.env._get_obs()
        
        # Convert to tensor for gradient computation
        state_tensor = torch.tensor(norm_state, dtype=torch.float32, requires_grad=True)
        
        # Compute gradient of action with respect to state
        action_tensor, _ = self.model.policy(state_tensor.unsqueeze(0))
        action_tensor.mean().backward()
        
        if state_tensor.grad is not None:
            gradient = state_tensor.grad.numpy()
            perturbation = self.epsilon * np.sign(gradient)
            
            # Apply perturbation to normalized state
            perturbed_norm_state = norm_state + perturbation
            perturbed_norm_state = np.clip(perturbed_norm_state, 0, 1)
            
            # Store perturbed state for observation
            self.env.last_perturbed_state = perturbed_norm_state
        else:
            self.env.last_perturbed_state = norm_state
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.env.last_perturbed_state = None
        obs, info = self.env.reset(**kwargs)
        return obs, info

class OLAttackWrapper(AttackWrapper):
    def __init__(self, env, model, epsilon=0.01):
        super(OLAttackWrapper, self).__init__(env, model, epsilon)
        self.attack_name = "OIA"
        
    def step(self, action):
        # Get current state for gradient computation
        norm_state = self.env._get_obs()
        
        # Convert to tensor for gradient computation
        state_tensor = torch.tensor(norm_state, dtype=torch.float32, requires_grad=True)
        
        # Compute gradient of value function with respect to state
        value = self.model.policy.value_net(state_tensor.unsqueeze(0))
        value.backward()
        
        if state_tensor.grad is not None:
            gradient = state_tensor.grad.numpy()
            perturbation = self.epsilon * np.sign(gradient)
            
            # Apply perturbation to normalized state
            perturbed_norm_state = norm_state + perturbation
            perturbed_norm_state = np.clip(perturbed_norm_state, 0, 1)
            
            # Store perturbed state for observation
            self.env.last_perturbed_state = perturbed_norm_state
        else:
            self.env.last_perturbed_state = norm_state
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.env.last_perturbed_state = None
        obs, info = self.env.reset(**kwargs)
        return obs, info

