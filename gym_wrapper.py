import gymnasium as gym
import numpy as np
from gymnasium import spaces
from Grid import MAPFEnvironment, Position

class CentralizedMAPFWrapper(gym.Env):
    """
    Wraps MAPFEnvironment for Centralized Multi-Agent PPO.
    Controls all agents simultaneously.
    """
    def __init__(self, env: MAPFEnvironment):
        super().__init__()
        self.env = env
        self.num_agents = len(env.starts)
        self.agent_ids = sorted(list(env.starts.keys()))
        
        # ACTION SPACE: MultiDiscrete([5, 5, 5, ...])
        # Each agent has 5 actions: 0=Wait, 1=Up, 2=Right, 3=Down, 4=Left
        self.action_space = spaces.MultiDiscrete([5] * self.num_agents)

        # OBSERVATION SPACE:
        # We use relative coordinates (dx, dy) to target and to other agents.
        # This is translation invariant and helps generalization.
        # For each agent:
        # [target_dx, target_dy] (2 values)
        # + [other_agent_dx, other_agent_dy] for each other agent (2 * (num_agents - 1) values)
        
        self.H = env.grid_map.height
        self.W = env.grid_map.width
        
        obs_per_agent = 2 + 2 * (self.num_agents - 1)
        obs_dim = self.num_agents * obs_per_agent
        
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(obs_dim,), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        return self._get_observation(), {}

    def step(self, action):
        # action is an array of ints, e.g. [1, 0, 4]
        joint_action = {self.agent_ids[i]: action[i] for i in range(self.num_agents)}
        
        prev_positions = self.env.agent_positions.copy()
        prev_finished = self.env.agent_finished.copy()
        
        _, done, info = self.env.step(joint_action)
        
        current_positions = self.env.agent_positions
        
        # REWARD FUNCTION
        reward = 0.0
        
        # Global penalty for time step (encourage speed)
        reward -= 0.1 * self.num_agents
        
        vertex_conflicts = info['vertex_conflicts']
        edge_conflicts = info['edge_conflicts']
        
        # Penalty for collisions (Increased)
        collision_penalty = -5.0
        reward += (len(vertex_conflicts) + len(edge_conflicts)) * collision_penalty
        
        for i, aid in enumerate(self.agent_ids):
            target = self.env.current_goals[aid]
            curr_pos = current_positions[aid]
            prev_pos = prev_positions[aid]
            
            # Calculate toroidal distances
            def dist(p1, p2):
                dr = abs(p1.row - p2.row)
                dc = abs(p1.col - p2.col)
                dr = min(dr, self.H - dr)
                dc = min(dc, self.W - dc)
                return dr + dc
            
            prev_dist = dist(prev_pos, target)
            curr_dist = dist(curr_pos, target)
            
            # Shaping reward: improvement in distance
            reward += (prev_dist - curr_dist) * 0.2  # Increased weight
            
            # Big reward for reaching a goal (sub-goal or final)
            # We detect this by checking if the target position changed (meaning we got a new goal)
            # OR if we are finished now but weren't before.
            
            prev_target = None # We don't have easy access to prev target without storing it.
            # But we can check if we are AT the target position.
            # If we are at target, we just completed a sub-goal (or final goal).
            # Note: Grid.py updates current_goals immediately after move.
            # So if we moved to a goal, current_goals[aid] is ALREADY the NEXT goal.
            # So we can't check `curr_pos == target`.
            
            # However, we can check `self.env.agent_goal_indices`.
            # We need to store prev indices.
            # Let's just rely on the "finished" flag for the big reward, 
            # and the distance shaping for the sub-goals.
            # Distance shaping naturally gives a reward when you step ON the goal 
            # (distance becomes 0, big drop from 1).
            # And when target switches, distance becomes large again, but that's next step.
            
            if self.env.agent_finished[aid] and not prev_finished[aid]:
                reward += 20.0 # Finished all goals
            
            # Penalty for invalid moves (hitting walls)
            if action[i] != 0 and curr_pos == prev_pos:
                 if not self.env.agent_finished[aid]:
                    reward -= 0.5

        truncated = info.get("time_limit_reached", False)
        
        return self._get_observation(), reward, done, truncated, info

    def _get_observation(self):
        obs = []
        for aid in self.agent_ids:
            pos = self.env.agent_positions[aid]
            target = self.env.current_goals[aid]
            
            # 1. Relative to Goal (Toroidal)
            dy = target.row - pos.row
            dx = target.col - pos.col
            
            # Wrap
            if dy > self.H / 2: dy -= self.H
            elif dy < -self.H / 2: dy += self.H
            
            if dx > self.W / 2: dx -= self.W
            elif dx < -self.W / 2: dx += self.W
            
            obs.extend([dy / self.H, dx / self.W])
            
            # 2. Relative to other agents
            for other_aid in self.agent_ids:
                if aid == other_aid: continue
                other_pos = self.env.agent_positions[other_aid]
                
                ody = other_pos.row - pos.row
                odx = other_pos.col - pos.col
                
                if ody > self.H / 2: ody -= self.H
                elif ody < -self.H / 2: ody += self.H
                
                if odx > self.W / 2: odx -= self.W
                elif odx < -self.W / 2: odx += self.W
                
                obs.extend([ody / self.H, odx / self.W])
                
        return np.array(obs, dtype=np.float32)