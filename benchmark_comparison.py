import time
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from Grid import GridMap, MAPFEnvironment, Position
from gym_wrapper import CentralizedMAPFWrapper
from AStar import find_multi_goal_paths
from train_marl import expand_goals_with_return
from AStar_visualize import visualize_paths

def run_cbs_benchmark(grid_data, starts, goals_list):
    print("\n--- Running CBS (A*) Benchmark ---")
    start_time = time.time()
    
    # CBS expects grid as list of lists (0/1)
    # Convert ascii grid to 0/1
    grid_01 = [[1 if c == '#' else 0 for c in row] for row in grid_data]
    
    solution = find_multi_goal_paths(starts, goals_list, grid_01, verbose=False)
    
    duration = time.time() - start_time
    
    if solution:
        makespan = max(len(p) for p in solution.values())
        total_steps = sum(len(p) for p in solution.values())
        print(f"CBS Solved in {duration:.4f}s")
        print(f"Makespan: {makespan}")
        print(f"Total Steps: {total_steps}")
        return {"success": True, "makespan": makespan, "steps": total_steps, "time": duration, "paths": solution}
    else:
        print("CBS Failed to find solution")
        return {"success": False, "makespan": 0, "steps": 0, "time": duration, "paths": {}}

def run_marl_benchmark(grid_map, starts_list, goals_list_raw, model_path="ppo_marl_agent"):
    print("\n--- Running MARL Benchmark ---")
    
    starts = {i: Position(r, c) for i, (r, c) in enumerate(starts_list)}
    full_goals_seq = expand_goals_with_return(starts_list, goals_list_raw)
    
    env_raw = MAPFEnvironment(grid_map, starts, full_goals_seq, max_time=200)
    env = CentralizedMAPFWrapper(env_raw)
    
    # Load Model
    try:
        model = PPO.load(model_path)
    except:
        print("Model not found. Please run train_marl.py first.")
        return {"success": False, "makespan": 0, "steps": 0, "time": 0, "paths": {}}
    
    obs, _ = env.reset()
    done = False
    steps = 0
    
    eval_start = time.time()
    
    paths = {aid: [starts_list[aid]] for aid in starts}
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        # Record positions
        for aid, pos in env.env.agent_positions.items():
            paths[aid].append((pos.row, pos.col))
            
        steps += 1
        if truncated:
            break
            
    eval_time = time.time() - eval_start
    
    success = info.get("all_at_goal", False)
    makespan = steps
    total_steps = sum(len(p) for p in paths.values())
    
    print(f"MARL Evaluation Time: {eval_time:.4f}s")
    print(f"Success: {success}")
    print(f"Makespan: {makespan}")
    print(f"Total Steps: {total_steps}")
    
    return {
        "success": success, 
        "makespan": makespan, 
        "steps": total_steps, 
        "time": eval_time,
        "paths": paths
    }

def main():
    # Problem Setup (Same as benchmark.py)
    grid_ascii = [
        "........",
        "........",
        "........",
        "........",
        "........",
        "........",
        "........",
        "........",
    ]
    
    starts_list = [(3, 1), (5, 3), (0, 5)]
    goals_list_raw = [
        [(5, 1), (4, 5), (6, 7)],  # Agent 0 
        [(6, 3), (1, 1), (7, 4)],  # Agent 1 
        [(1, 5), (7, 1), (5, 3)],  # Agent 2
    ]
    
    # Run CBS
    cbs_results = run_cbs_benchmark(grid_ascii, starts_list, goals_list_raw)
    
    # Run MARL
    grid_map = GridMap.from_ascii(grid_ascii)
    marl_results = run_marl_benchmark(grid_map, starts_list, goals_list_raw)
    
    # Plot Comparison
    labels = ['CBS (Optimal)', 'MARL (PPO)']
    makespans = [cbs_results['makespan'], marl_results['makespan']]
    times = [cbs_results['time'], marl_results['time']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Makespan
    ax1.bar(labels, makespans, color=['blue', 'orange'])
    ax1.set_title('Makespan (Lower is Better)')
    ax1.set_ylabel('Steps')
    for i, v in enumerate(makespans):
        ax1.text(i, v + 1, str(v), ha='center')
        
    # Execution Time
    ax2.bar(labels, times, color=['blue', 'orange'])
    ax2.set_title('Execution Time (Lower is Better)')
    ax2.set_ylabel('Seconds')
    for i, v in enumerate(times):
        ax2.text(i, v, f"{v:.4f}s", ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig('comparison_result.png')
    print("Comparison plot saved to 'comparison_result.png'")
    # plt.show()
    
    # Visualize Paths
    grid_01 = [[1 if c == '#' else 0 for c in row] for row in grid_ascii]
    
    if cbs_results['success']:
        print("Visualizing CBS Solution (Close window to continue)...")
        visualize_paths(grid_01, starts_list, goals_list_raw, cbs_results['paths'])
        
    if marl_results['success'] or marl_results['steps'] > 0:
        print(f"Visualizing MARL Solution (Path lengths: {[len(p) for p in marl_results['paths'].values()]})")
        visualize_paths(grid_01, starts_list, goals_list_raw, marl_results['paths'])

if __name__ == "__main__":
    main()