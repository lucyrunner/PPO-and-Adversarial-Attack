#!/usr/bin/env python
# coding: utf-8

# In[23]:


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class ACCEnv(gym.Env):
    def __init__(self, dt=0.1, max_steps=500):
        super(ACCEnv, self).__init__()
        
        # Parameters from the paper
        self.dt = dt
        self.max_steps = max_steps
        self.current_step = 0
        
        # Vehicle parameters
        self.ego_pos = 0.0
        self.ego_vel = 5.0  # Initial speed as in paper
        self.lead_vel = 15.0  # Constant lead speed during training
        self.lead_pos = 20.0  # Initial lead position
        
        # Acceleration limits
        self.max_accel = 2.0
        self.max_decel = -3.5
        
        # Safety parameters
        self.T_h = 1.5  # Time headway
        self.d0 = 5.0   # Standstill distance
        
        # Target speed
        self.v_ref = 15.0
        self.v_min = 10.0
        self.v_max = 20.0
        
        # Action space: continuous acceleration
        self.action_space = spaces.Box(
            low=np.array([self.max_decel], dtype=np.float32), 
            high=np.array([self.max_accel], dtype=np.float32), 
            shape=(1,), 
            dtype=np.float32
        )
        
        # State space: [relative_distance, relative_velocity, ego_velocity, iteration_count]
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 0, 0], dtype=np.float32), 
            high=np.array([100, 10, 30, max_steps], dtype=np.float32), 
            shape=(4,),
            dtype=np.float32
        )
        
        # For testing scenarios
        self.test_mode = False
        self.brake_start_time = None
        self.brake_duration = 3.0
        self.lead_decel = -2.0
        
        # For rendering
        self.fig = None
        self.axs = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.ego_pos = 0.0
        self.ego_vel = 5.0  # Non-zero initial speed as in paper
        
        # Randomize lead vehicle initial conditions for diversity
        if seed is not None:
            np.random.seed(seed)
        self.lead_pos = np.random.uniform(15, 25)
        self.lead_vel = np.random.uniform(12, 18)
        
        if self.test_mode:
            self.lead_vel = 15.0  # Fixed for consistent testing
            self.lead_pos = 20.0
            self.brake_start_time = 100  # Start braking at step 100
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Get normalized observation as described in paper"""
        rel_dist = self.lead_pos - self.ego_pos
        rel_vel = self.lead_vel - self.ego_vel
        
        # Normalize state to [0, 1] range for stable training
        norm_rel_dist = rel_dist / 100.0
        norm_rel_vel = (rel_vel + 10) / 20.0  # Map [-10, 10] to [0, 1]
        norm_ego_vel = self.ego_vel / 30.0
        norm_step = self.current_step / self.max_steps
        
        return np.array([norm_rel_dist, norm_rel_vel, norm_ego_vel, norm_step], dtype=np.float32)
    
    def _get_denormalized_state(self):
        """Get actual physical state values"""
        rel_dist = self.lead_pos - self.ego_pos
        rel_vel = self.lead_vel - self.ego_vel
        return np.array([rel_dist, rel_vel, self.ego_vel, self.current_step])
    
    def step(self, action):
        action = np.clip(action, self.max_decel, self.max_accel)[0]
        
        # Apply safety filter (CBF)
        safe_action = self._safety_filter(action)
        
        # Update ego vehicle
        self.ego_vel += safe_action * self.dt
        self.ego_vel = np.clip(self.ego_vel, 0, 30)  # Physical limits
        self.ego_pos += self.ego_vel * self.dt
        
        # Update lead vehicle
        if self.test_mode and self.brake_start_time is not None:
            if self.current_step >= self.brake_start_time and \
               self.current_step < self.brake_start_time + self.brake_duration/self.dt:
                self.lead_vel = max(0, self.lead_vel + self.lead_decel * self.dt)
            elif self.current_step >= self.brake_start_time + self.brake_duration/self.dt:
                self.lead_vel = max(0, 15.0 + self.lead_decel * self.brake_duration)  # New speed
        
        self.lead_pos += self.lead_vel * self.dt
        
        # Calculate reward
        reward = self._calculate_reward(safe_action)
        
        # Check termination conditions
        terminated = False
        truncated = False
        collision = (self.lead_pos - self.ego_pos) <= 0
        timeout = self.current_step >= self.max_steps
        reached_end = self.ego_pos >= 1000  # Arbitrary road length
        
        if collision:
            reward -= 50  # Large penalty for collision
            terminated = True
        elif reached_end:
            reward += 50  # Large reward for completing episode
            terminated = True
        elif timeout:
            truncated = True
            
        self.current_step += 1
        
        return self._get_obs(), reward, terminated, truncated, {
            'collision': collision,
            'safe_action': safe_action,
            'original_action': action
        }
    
    def _calculate_reward(self, action):
        """Calculate reward according to paper's formulation"""
        rel_dist = self.lead_pos - self.ego_pos
        
        # Step penalty (encourage faster completion)
        r_step = -0.05
        
        # Speed reward (piecewise linear as in paper)
        if self.ego_vel < self.v_min:
            r_speed = -0.1 * (self.v_min - self.ego_vel)
        elif self.ego_vel > self.v_max:
            r_speed = -0.1 * (self.ego_vel - self.v_max)
        else:
            r_speed = 0.1 * (self.ego_vel - self.v_min)
        
        # Safety distance penalty
        safe_dist = self.d0 + self.T_h * self.ego_vel
        if rel_dist < safe_dist:
            r_safe = -2.0 * (safe_dist - rel_dist) ** 2
        else:
            r_safe = 0.0
            
        # Action penalty (for comfort)
        r_action = -0.01 * (action ** 2)
        
        # Idling penalty (if standing still for too long)
        r_idling = -20.0 if self.ego_vel < 0.1 and self.current_step > 10 else 0.0
        
        total_reward = r_step + r_speed + r_safe + r_action + r_idling
        return total_reward
    
    def _safety_filter(self, action):
        """CBF safety filter implementation"""
        rel_dist = self.lead_pos - self.ego_pos
        rel_vel = self.lead_vel - self.ego_vel
        
        # Calculate maximum safe acceleration using CBF constraint
        # Simplified version assuming lead acceleration = 0
        h = rel_dist - self.T_h * self.ego_vel
        
        if h < 0:  # Already unsafe, emergency braking
            return self.max_decel
            
        # Calculate maximum allowed acceleration
        max_safe_accel = (h + rel_vel * self.dt) / (self.T_h * self.dt)
        
        # Apply constraint
        safe_action = min(action, max_safe_accel)
        return np.clip(safe_action, self.max_decel, self.max_accel)
    
    def set_test_mode(self, enabled=True):
        self.test_mode = enabled
        
    def render(self):
        if self.fig is None:
            self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
            plt.ion()
            
        rel_dist = self.lead_pos - self.ego_pos
        
        self.axs[0,0].clear()
        self.axs[0,0].plot(self.ego_pos, 0, 'bo', markersize=10, label='Ego')
        self.axs[0,0].plot(self.lead_pos, 0, 'ro', markersize=10, label='Lead')
        self.axs[0,0].set_xlim(max(0, self.ego_pos-10), self.lead_pos+10)
        self.axs[0,0].set_title('Vehicle Positions')
        self.axs[0,0].legend()
        
        self.axs[0,1].clear()
        self.axs[0,1].plot(self.current_step, self.ego_vel, 'bo', label='Ego')
        self.axs[0,1].plot(self.current_step, self.lead_vel, 'ro', label='Lead')
        self.axs[0,1].set_title('Velocities')
        self.axs[0,1].set_ylabel('Speed (m/s)')
        self.axs[0,1].legend()
        
        self.axs[1,0].clear()
        self.axs[1,0].plot(self.current_step, rel_dist, 'go', label='Distance')
        safe_dist = self.d0 + self.T_h * self.ego_vel
        self.axs[1,0].axhline(y=safe_dist, color='r', linestyle='--', label='Safe Distance')
        self.axs[1,0].set_title('Relative Distance')
        self.axs[1,0].set_ylabel('Distance (m)')
        self.axs[1,0].legend()
        
        self.axs[1,1].clear()
        self.axs[1,1].plot(self.current_step, self.ego_vel - self.lead_vel, 'purple', label='Rel Velocity')
        self.axs[1,1].set_title('Relative Velocity')
        self.axs[1,1].set_ylabel('Velocity Diff (m/s)')
        self.axs[1,1].legend()
        
        plt.tight_layout()
        plt.pause(0.01)
        
        return None
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axs = None

