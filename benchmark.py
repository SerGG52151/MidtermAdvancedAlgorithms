from __future__ import annotations

import time
from typing import Dict

from Grid import GridMap, Position
from AStar import run_cbs_on_map, visualize_solution


def run_example_three_agent() -> None:
    ascii_map = [
        "#########",
        "#S.A.B.G#",
        "#.......#",
        "#G.B.A.S#",
        "#########",
    ]

    grid = GridMap.from_ascii(ascii_map)

    starts: Dict[int, Position] = {
        0: Position(1, 1),  # S
        1: Position(1, 3),  # A
        2: Position(1, 5),  # B
    }

    goals: Dict[int, Position] = {
        0: Position(3, 7),
        1: Position(3, 5),
        2: Position(3, 3),
    }

    t0 = time.time()
    solution = run_cbs_on_map(grid, starts, goals, max_time=100)
    dt = time.time() - t0

    if solution is None:
        print("No solution found")
        return

    print(f"Solution found in {dt:.3f}s")
    max_len = max(len(p) for p in solution.values())
    for a in sorted(solution.keys()):
        path = solution[a]
        print(f"Agent {a}: {[(p.row,p.col) for p in path]}")

    print(f"Makespan (max path length): {max_len}")
    # visualize and save the solution plot
    try:
        visualize_solution(grid, solution, starts=starts, goals=goals)
    except Exception as e:
        print(f"Visualization failed: {e}")


if __name__ == "__main__":
    run_example_three_agent()
