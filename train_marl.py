import gymnasium as gym
from stable_baselines3 import PPO
from Grid import GridMap, MAPFEnvironment, Position
from gym_wrapper import CentralizedMAPFWrapper
import os

def expand_goals_with_return(starts, goals_list):
    """
    Expands the goal list to include the return-to-start trips.
    Input: [[g1, g2], ...]
    Output: [[g1, start, g2, start], ...]
    """
    expanded = {}
    for i, goals in enumerate(goals_list):
        agent_seq = []
        start = starts[i]
        start_pos = Position(start[0], start[1])
        for g in goals:
            agent_seq.append(Position(g[0], g[1]))
            agent_seq.append(start_pos)
        expanded[i] = agent_seq
    return expanded

def main():
    # 1. Setup the Benchmark Scenario (Same as benchmark.py)
    grid_data = [
        "........",
        "........",
        "........",
        "........",
        "........",
        "........",
        "........",
        "........",
    ]
    grid = GridMap.from_ascii(grid_data)
    
    # Starts and Goals from benchmark.py
    starts_list = [(3, 1), (5, 3), (0, 5)]
    goals_list_raw = [
        [(5, 1), (4, 5)],  # Agent 0
        [(6, 3), (1, 1)],  # Agent 1
        [(1, 5), (7, 1)],  # Agent 2
    ]
    
    starts = {i: Position(r, c) for i, (r, c) in enumerate(starts_list)}
    
    # Expand goals to include return trips (Service -> Home -> Service -> Home)
    full_goals_seq = expand_goals_with_return(starts_list, goals_list_raw)

    # 2. Initialize Environment
    env_raw = MAPFEnvironment(grid, starts, full_goals_seq, max_time=200)
    env = CentralizedMAPFWrapper(env_raw)

    # 3. Create PPO Model
    # Using MlpPolicy because our observation is a flat vector of coordinates
    # Added tensorboard_log="./ppo_marl_logs/" to visualize training progress
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, ent_coef=0.03, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_marl_logs/")

    print("--- Starting MARL Training (Centralized PPO) ---")
    
    # Train for more steps to ensure convergence on this multi-agent task
    # Increased to 2,000,000 steps for better convergence
    model.learn(total_timesteps=2000000)
    
    print("--- Training Finished ---")

    # 4. Save Model
    model.save("ppo_marl_agent_2")
    print("Model saved as 'ppo_marl_agent_2.zip'")

if __name__ == "__main__":
    main()