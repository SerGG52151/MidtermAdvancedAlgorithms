from __future__ import annotations

import time
from typing import Dict, List

from AStar import find_multi_goal_paths
from AStar_visualize import visualize_paths


def run_example_three_services() -> None:
    """Demonstrates the Three Services Problem with CBS.
    
    Three agents must each visit service locations in sequence:
    - Agent 0: starts at (1,1), visits services at (2,3) and (2,5), ends at (3,7)
    - Agent 1: starts at (1,3), visits services at (2,5) and (2,1), ends at (3,5)
    - Agent 2: starts at (1,5), visits services at (2,1) and (2,3), ends at (3,3)
    
    This creates interesting conflicts as agents need to visit each other's
    intermediate locations, requiring CBS to coordinate their movements.
    """
    # Define a simple 8x8 open grid (0 = free, 1 = obstacle)
    grid = [
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
    ]

    # Define starts at the top in a line and goals at the bottom in a line
    # Agents start at row 0, goals at row 7 (bottom)
    starts = [(3, 1), (5, 3), (0, 5)]  # Top row, spread across
    goals_list = [
        [(5, 1), (4, 5), (6, 7)],  # Agent 0: bottom left, center, right
        [(6, 3), (1, 1), (7, 4)],  # Agent 1: bottom center, right, left
        [(1, 5), (7, 1), (5, 3)],  # Agent 2: bottom right, left, center
    ]
    goals_listOG = [
        [(7, 1), (7, 3), (7, 5)],  # Agent 0: bottom left, center, right
        [(7, 3), (7, 5), (7, 1)],  # Agent 1: bottom center, right, left
        [(7, 5), (7, 1), (7, 3)],  # Agent 2: bottom right, left, center
    ]

    print("=" * 60)
    print("THREE AGENTS - Multi-Goal Conflict-Based Search (CBS)")
    print("=" * 60)
    print("\nProblem setup:")
    for i, (start, goals) in enumerate(zip(starts, goals_list)):
        print(f"  Agent {i}: Start {start} -> Goals {goals}")
    print()

    t0 = time.time()
    solution = find_multi_goal_paths(starts, goals_list, grid, verbose=True)
    dt = time.time() - t0

    if solution is None:
        print("❌ No multi-goal solution found")
        return

    print(f"✓ Solution found in {dt:.3f}s\n")
    max_len = max(len(p) for p in solution.values())
    total_cost = sum(len(p) for p in solution.values())

    print("Agent paths (each service trip shown separately):")
    for agent_id in sorted(solution.keys()):
        path = solution[agent_id]
        start_pos = starts[agent_id]
        agent_goals = goals_list[agent_id]
        
        print(f"\n  Agent {agent_id} (total {len(path)} steps):")
        
        # Split path into segments for each service
        # Each round trip: start -> goal -> start
        current_idx = 0
        for service_num, goal in enumerate(agent_goals, 1):
            # Find where this goal appears in the path (first occurrence after current_idx)
            goal_idx = None
            for i in range(current_idx, len(path)):
                if path[i] == goal:
                    goal_idx = i
                    break
            
            if goal_idx is not None:
                # Extract outbound journey only (from start_pos to goal)
                # Find where start_pos appears at or before current_idx
                segment_start = current_idx
                for j in range(current_idx, -1, -1):
                    if path[j] == start_pos:
                        segment_start = j
                        break
                
                # Outbound segment: from start to goal
                outbound = path[segment_start:goal_idx+1]
                print(f"    Service {service_num} to {goal}: {outbound}")
                
                # Move past this round trip (goal back to start)
                # Find next occurrence of start_pos after goal
                for i in range(goal_idx + 1, len(path)):
                    if path[i] == start_pos:
                        current_idx = i
                        break
                else:
                    current_idx = len(path)

    print(f"\nMetrics:")
    print(f"  Makespan (max path length): {max_len}")
    print(f"  Sum of costs: {total_cost}")
    print(f"  Average path length: {total_cost / len(solution):.1f}")
    print()

    # Visualize the solution - pass goals_list for proper segment separation
    visualize_paths(grid, starts, goals_list, solution)


if __name__ == "__main__":
    run_example_three_services()
