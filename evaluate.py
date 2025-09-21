from __future__ import annotations
import argparse, os, csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from acc_env import ACCEnv
from attacks import FGSMAttack, OIAttack

def make_env(brake_profile=True, normalize_obs=True):
    def _thunk():
        # Evaluation uses a braking scenario to test reactions
        return ACCEnv(brake_profile=brake_profile, normalize_obs=normalize_obs)
    return _thunk

def load_model_and_env(logdir: str):
    env = DummyVecEnv([make_env(brake_profile=True, normalize_obs=True)])
    env = VecNormalize.load(os.path.join(logdir, 'vecnormalize.pkl'), env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(os.path.join(logdir, 'ppo_acc'))
    return model, env

def run_episode(model, env, attack=None, eps=0.01, render_traj=False):
    obs = env.reset()[0]
    done = False
    traj = {k: [] for k in ['Δx','v','a','r']}
    total_r = 0.0
    collisions = 0
    prev_a = 0.0
    rmse_accum = 0.0
    rmse_count = 0

    # Wrap model with attack if requested
    atk = None
    if attack == 'fgsm':
        atk = FGSMAttack(model, epsilon=eps, device='cpu')
    elif attack == 'oia':
        atk = OIAttack(model, epsilon=eps, device='cpu')

    while True:
        if atk is None:
            action, _ = model.predict(obs, deterministic=True)
            obs_in = obs
        else:
            action, obs_in = atk.act(obs)

        obs, reward, term, trunc, info = env.step(action)
        total_r += reward[0] if isinstance(reward, np.ndarray) else reward
        traj['Δx'].append(info[0]['Δx'] if isinstance(info, list) else info['Δx'])
        traj['v'].append(info[0]['v'] if isinstance(info, list) else info['v'])
        traj['a'].append(info[0]['a'] if isinstance(info, list) else info['a'])
        traj['r'].append(reward[0] if isinstance(reward, np.ndarray) else reward)

        # For stealth metric: RMSE between obs_in and obs (both normalized)
        if atk is not None:
            diff = (obs_in - obs)
            rmse_accum += float((diff**2).mean())
            rmse_count += 1

        done = bool(term) or bool(trunc)
        if term:
            collisions = 1
        if done:
            break

    # Metrics
    jerk = np.mean(np.abs(np.diff(traj['a']))) if len(traj['a']) > 1 else 0.0
    rmse = np.sqrt(rmse_accum / max(1, rmse_count))
    return {
        'return': total_r,
        'collision': collisions,
        'jerk': jerk,
        'rmse': rmse,
        'traj': traj
    }

def plot_traj(traj, title, out_png):
    t = np.arange(len(traj['Δx']))
    plt.figure()
    plt.plot(t, traj['Δx'])
    plt.xlabel('t (steps)'); plt.ylabel('Δx (m)'); plt.title(title + ' — headway (Δx)')
    plt.savefig(out_png.replace('.png','_dx.png'), bbox_inches='tight'); plt.close()

    t = np.arange(len(traj['v']))
    plt.figure()
    plt.plot(t, traj['v'])
    plt.xlabel('t (steps)'); plt.ylabel('v (m/s)'); plt.title(title + ' — ego speed (v)')
    plt.savefig(out_png.replace('.png','_v.png'), bbox_inches='tight'); plt.close()

    t = np.arange(len(traj['a']))
    plt.figure()
    plt.plot(t, traj['a'])
    plt.xlabel('t (steps)'); plt.ylabel('a (m/s^2)'); plt.title(title + ' — acceleration (a)')
    plt.savefig(out_png.replace('.png','_a.png'), bbox_inches='tight'); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='runs/ppo_baseline')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--attack', type=str, default='none', choices=['none','fgsm','oia'])
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--compare', action='store_true', help='Run baseline vs FGSM vs OIA')
    args = parser.parse_args()

    os.makedirs('artifacts', exist_ok=True)

    model, env = load_model_and_env(args.logdir)

    def eval_many(which: str | None):
        atk = 'none' if which is None else which
        rets, cols, jerks, rmses = [], [], [], []
        sample_traj = None
        for ep in range(args.episodes):
            res = run_episode(model, env, attack=which, eps=args.eps)
            rets.append(res['return'])
            cols.append(res['collision'])
            jerks.append(res['jerk'])
            rmses.append(res['rmse'])
            if sample_traj is None:
                sample_traj = res['traj']
        avg = {
            'avg_return': float(np.mean(rets)),
            'collision_rate': float(np.mean(cols)),
            'avg_jerk': float(np.mean(jerks)),
            'avg_rmse': float(np.mean(rmses)),
        }
        # Save CSV
        out_csv = f"artifacts/metrics_{atk}.csv"
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['metric','value'])
            for k,v in avg.items():
                w.writerow([k,v])
        # Plots for sample traj
        plot_traj(sample_traj, f'{atk.upper()} sample episode', f'artifacts/{atk}_traj.png')
        print(f"{atk}: {avg}")
        return avg

    if args.compare:
        base = eval_many(None)
        fgsm = eval_many('fgsm')
        oia  = eval_many('oia')
        # Simple print summary
        print('--- Comparison ---')
        print('Baseline:', base)
        print('FGSM    :', fgsm)
        print('OIA     :', oia)
    else:
        which = None if args.attack=='none' else args.attack
        eval_many(which)

if __name__ == '__main__':
    main()
