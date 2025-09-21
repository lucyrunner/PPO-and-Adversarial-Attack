from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ACCEnv(gym.Env):
    """
    1D ACC environment with simple kinematics and a headway-based safety filter.

    Observation s = [Δx, Δv, v]  (raw units: m, m/s, m/s)
    Action a = acceleration in m/s^2 (continuous).

    Normalization:
      - If normalize_obs=True, env outputs normalized obs in [-1, 1] using fixed ranges.
      - Attack budgets (epsilon) should be defined in normalized space.
    """
    metadata = {"render.modes": []}

    def __init__(
        self,
        dt: float = 0.1,
        v_ref: float = 15.0,
        a_min: float = -3.5,
        a_max: float = 2.0,
        Th: float = 1.5,
        d0: float = 5.0,
        w_v: float = 0.5,
        w_s: float = 2.0,
        w_a: float = 0.01,
        lead_v0: float = 15.0,
        brake_profile: bool = False,
        brake_start_s: float = 5.0,
        brake_dur_s: float = 3.0,
        lead_decel: float = -2.0,
        episode_seconds: float = 20.0,
        seed: int | None = None,
        normalize_obs: bool = True,
        obs_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.v_ref = v_ref
        self.a_min = a_min
        self.a_max = a_max
        self.Th = Th
        self.d0 = d0
        self.w_v = w_v
        self.w_s = w_s
        self.w_a = w_a
        self.lead_v0 = lead_v0
        self.brake_profile = brake_profile
        self.brake_start_s = brake_start_s
        self.brake_dur_s = brake_dur_s
        self.lead_decel = lead_decel
        self.episode_steps = int(episode_seconds / dt)
        self.normalize_obs = normalize_obs
        self.obs_clip = obs_clip

        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Observation space (normalized if normalize_obs=True)
        high = np.array([1.0, 1.0, 1.0], dtype=np.float32) if normalize_obs else np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Action space in raw units (m/s^2)
        self.action_space = spaces.Box(
            low=np.array([self.a_min], dtype=np.float32),
            high=np.array([self.a_max], dtype=np.float32),
            dtype=np.float32,
        )

        # Normalization ranges for obs (hand-tuned for ACC task)
        # Δx in [0, 200] m -> map to [-1, 1]
        # Δv in [-20, 20] m/s -> map to [-1, 1]
        # v in [0, 30] m/s -> map to [-1, 1]
        self._x_range = (0.0, 200.0)
        self._dv_range = (-20.0, 20.0)
        self._v_range = (0.0, 30.0)

        self.reset()

    # ---------- Normalization helpers ----------
    def _norm(self, s_raw: np.ndarray) -> np.ndarray:
        Δx, Δv, v = s_raw
        def _scale(val, lo, hi):
            # map [lo,hi] -> [-1,1]
            val = (val - lo) / (hi - lo + 1e-8)
            return np.clip(2.0*val - 1.0, -self.obs_clip, self.obs_clip)
        return np.array([
            _scale(Δx, *self._x_range),
            _scale(Δv, *self._dv_range),
            _scale(v, *self._v_range),
        ], dtype=np.float32)

    def _denorm(self, s_norm: np.ndarray) -> np.ndarray:
        # Map [-1,1] back to raw
        def _inv(vn, lo, hi):
            x = (vn + 1.0) * 0.5 * (hi - lo) + lo
            return x
        Δx = _inv(s_norm[0], *self._x_range)
        Δv = _inv(s_norm[1], *self._dv_range)
        v  = _inv(s_norm[2], *self._v_range)
        return np.array([Δx, Δv, v], dtype=np.float32)

    # ---------- Safety filter ----------
    def _amax_safe(self, Δx: float, Δv: float, v: float) -> float:
        # Eq.(5): a_max_safe = (Δx - Th*v + Δv*dt) / (Th*dt)
        return (Δx - self.Th * v + Δv * self.dt) / (self.Th * self.dt + 1e-8)

    def _apply_safety(self, a_rl: float, Δx: float, Δv: float, v: float) -> float:
        a_safe_max = self._amax_safe(Δx, Δv, v)
        a_clamped = min(a_rl, a_safe_max)
        return float(np.clip(a_clamped, self.a_min, self.a_max))

    # ---------- Gym API ----------
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        # Ego starts behind lead with ~30-50 m headway and near target speed with small noise
        self.x_e = 0.0
        self.v_e = float(np.clip(self.v_ref + self.np_random.normal(0, 0.5), 0.0, self._v_range[1]))
        self.a_prev = 0.0

        self.x_l = float(self.np_random.uniform(30.0, 50.0))
        self.v_l = self.lead_v0
        self._t = 0
        self._collision = False

        obs = self._get_obs()
        info = {}
        return obs, info

    def _lead_step(self):
        t = self._t * self.dt
        if self.brake_profile and (t >= self.brake_start_s) and (t < self.brake_start_s + self.brake_dur_s):
            self.v_l = max(self.v_l + self.lead_decel * self.dt, 0.0)
        # else constant

        self.x_l = self.x_l + self.v_l * self.dt

    def _get_obs(self) -> np.ndarray:
        Δx = self.x_l - self.x_e
        Δv = self.v_l - self.v_e
        s_raw = np.array([Δx, Δv, self.v_e], dtype=np.float32)
        if self.normalize_obs:
            return self._norm(s_raw)
        return s_raw

    def _reward(self) -> float:
        Δx = self.x_l - self.x_e
        Δv = self.v_l - self.v_e
        v  = self.v_e
        d_safe = self.d0 + self.Th * v
        r_speed = - self.w_v * (v - self.v_ref)**2
        r_safe  = - self.w_s * max(0.0, d_safe - Δx)**2
        r_act   = - self.w_a * (self.a_prev**2)
        return float(r_speed + r_safe + r_act)

    def step(self, action):
        a_rl = float(np.clip(action, self.a_min, self.a_max)[0])
        # Convert (possibly normalized obs) to raw for safety filter
        s = self._get_obs()
        if self.normalize_obs:
            s_raw = self._denorm(s)
        else:
            s_raw = s
        Δx, Δv, v = float(s_raw[0]), float(s_raw[1]), float(s_raw[2])

        a = self._apply_safety(a_rl, Δx, Δv, v)

        # Ego dynamics
        self.x_e = self.x_e + self.v_e * self.dt
        self.v_e = max(self.v_e + a * self.dt, 0.0)
        self.a_prev = a

        # Lead dynamics
        self._lead_step()

        # Check collision
        if (self.x_l - self.x_e) <= 0.0:
            self._collision = True

        obs = self._get_obs()
        reward = self._reward()
        self._t += 1
        terminated = self._collision
        truncated = self._t >= self.episode_steps
        info = {"collision": self._collision, "a": a, "v": self.v_e, "Δx": self.x_l - self.x_e}
        return obs, reward, terminated, truncated, info
