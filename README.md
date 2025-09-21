# PPO + Adversarial Attacks (FGSM & OIA) in 1D ACC Environment

This repo provides a **minimal, reproducible** codebase to train a PPO agent for a simplified Adaptive Cruise Control (ACC) task and evaluate two adversarial attacks:
**FGSM** (Fast Gradient Sign Method) and **OIA** (Optimism Induction Attack).

It follows the environment, reward, safety filter (CBF), and evaluation ideas summarized in the attached notes.

## Environment Highlights
- 1D car-following: ego (agent) follows a lead vehicle on a straight road.
- State: `[Δx, Δv, v]` where `Δx = x_lead - x_ego`, `Δv = v_lead - v_ego`, `v = v_ego`.
- Action: ego acceleration `a` (continuous, clipped to [a_min, a_max]).
- Reward: speed tracking + safe distance + action penalty.
- **Safety filter (CBF)**: clamps unsafe accelerations using a headway-based barrier.

## Attacks
- **FGSM**: Perturbs observation in the direction increasing the policy mean action.
- **OIA**: Perturbs observation to *increase the value function* (induces optimism).

## Quickstart

```bash
# 1) Create and activate a Python 3.10+ virtual environment
python -m venv .venv
source .venv/bin/activate           # on Windows: .venv\Scripts\activate

# 2) Install requirements
pip install -r requirements.txt

# 3) (Optional) Train
python train.py --total-steps 300000 --logdir runs/ppo_baseline

# 4) Evaluate baseline and attacks (will load last model unless specified)
python evaluate.py --episodes 20 --attack none      # baseline
python evaluate.py --episodes 20 --attack fgsm
python evaluate.py --episodes 20 --attack oia

# 5) Produce a comparison run (baseline vs FGSM vs OIA)
python evaluate.py --episodes 50 --compare
```

All plots and CSV metrics are saved to `artifacts/`.

## Files
- `acc_env.py`: Gymnasium environment + safety filter + normalization helpers.
- `attacks.py`: FGSM & OIA wrappers for SB3 policies (PyTorch autograd).
- `train.py`: PPO training with Stable-Baselines3 + VecNormalize.
- `evaluate.py`: Evaluation, metrics (collision rate, return, jerk, RMSE), plots.
- `requirements.txt`: minimal dependencies.

> Note: You may adjust reward weights and safety parameters in `acc_env.py`.
