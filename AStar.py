from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple, Set

from Grid import (
    GridMap,
    Position,
    Constraint,
    Conflict,
    detect_conflicts,
    MAPFEnvironment,
)
import warnings

try:
    # prefer to use GridMap.draw, but provide fallback if matplotlib isn't available
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional plotting
    plt = None


def manhattan(a: Position, b: Position) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


def astar_single_agent(
    env: MAPFEnvironment,
    agent_id: int,
    start: Position,
    goal: Position,
    constraints: Optional[List[Constraint]] = None,
    max_time: int = 50,
) -> Optional[List[Position]]:
    """Time-expanded A* for a single agent respecting constraints.

    Returns a list of Positions indexed by time (t=0..T). Returns None if no path.
    """

    start_key = (start, 0)

    # Use an incremental tie-breaker to ensure heap tuples are always comparable
    open_heap: List[Tuple[int, int, int, Position, int]] = []  # (f, g, counter, pos, time)
    push_counter = 0
    heapq.heappush(open_heap, (manhattan(start, goal), 0, push_counter, start, 0))
    push_counter += 1

    came_from: Dict[Tuple[Position, int], Tuple[Position, int]] = {}
    g_score: Dict[Tuple[Position, int], int] = {start_key: 0}

    while open_heap:
        f, g, _, pos, time = heapq.heappop(open_heap)

        if pos == goal:
            # reconstruct path
            path: List[Position] = [pos]
            cur = (pos, time)
            while cur in came_from:
                prev = came_from[cur]
                path.append(prev[0])
                cur = prev
            path.reverse()
            return path

        if time >= max_time:
            continue

        successors = env.successors_for_agent(agent_id, pos, time, constraints)
        for nxt in successors:
            next_time = time + 1
            neighbor_key = (nxt, next_time)
            tentative_g = g + 1
            if tentative_g < g_score.get(neighbor_key, 10**9):
                g_score[neighbor_key] = tentative_g
                came_from[neighbor_key] = (pos, time)
                h = manhattan(nxt, goal)
                heapq.heappush(open_heap, (tentative_g + h, tentative_g, push_counter, nxt, next_time))
                push_counter += 1

    return None


class CBSNode:
    def __init__(self, constraints: Optional[List[Constraint]] = None, paths: Optional[Dict[int, List[Position]]] = None):
        self.constraints: List[Constraint] = list(constraints or [])
        self.paths: Dict[int, List[Position]] = dict(paths or {})

    @property
    def cost(self) -> int:
        # Sum of individual path lengths (can be used as priority)
        return sum(len(p) for p in self.paths.values())


def cbs(
    env: MAPFEnvironment,
    max_time: int = 50,
) -> Optional[Dict[int, List[Position]]]:
    """Conflict-Based Search for MAPF. Works for small agent counts (e.g., 3 agents).

    Returns a dictionary mapping agent_id -> path (list of Positions by time)
    or None if no solution found within `max_time`.
    """

    # initial paths (no constraints)
    root = CBSNode()
    for a, s in env.starts.items():
        path = astar_single_agent(env, a, s, env.goals[a], None, max_time)
        if path is None:
            return None
        root.paths[a] = path

    open_list: List[Tuple[int, int, CBSNode]] = []
    heapq.heappush(open_list, (root.cost, 0, root))
    node_counter = 1

    while open_list:
        _, _, node = heapq.heappop(open_list)

        conflicts = detect_conflicts(node.paths)
        if not conflicts:
            return node.paths

        conflict = conflicts[0]

        for idx in range(2):
            agent = conflict.agents[idx]
            child = CBSNode(constraints=node.constraints, paths=node.paths)

            # build constraint for this agent
            if conflict.kind == "vertex":
                c = Constraint(agent_id=agent, time=conflict.time, position=conflict.pos1)
            else:  # edge conflict
                # conflict.pos1/pos2 correspond to the first agent in the pair
                if idx == 0:
                    from_p, to_p = conflict.pos1, conflict.pos2
                else:
                    # invert for the second agent
                    from_p, to_p = conflict.pos2, conflict.pos1
                c = Constraint(agent_id=agent, time=conflict.time, from_position=from_p, to_position=to_p)

            child.constraints = list(node.constraints) + [c]

            # replan only for the constrained agent
            # pass full constraint list; astar will ignore constraints for other agents
            path = astar_single_agent(env, agent, env.starts[agent], env.goals[agent], child.constraints, max_time)
            if path is None:
                continue
            child.paths = dict(node.paths)
            child.paths[agent] = path

            heapq.heappush(open_list, (child.cost, node_counter, child))
            node_counter += 1

    return None


def run_cbs_on_map(
    grid_map: GridMap,
    starts: Dict[int, Position],
    goals: Dict[int, Position],
    max_time: int = 50,
) -> Optional[Dict[int, List[Position]]]:
    """Convenience wrapper: build an environment and run CBS on it.

    Returns the dictionary agent_id->path or None if no solution.
    """
    env = MAPFEnvironment(grid_map, starts, goals, max_time=max_time)
    return cbs(env, max_time=max_time)


def visualize_solution(
    grid_map: GridMap,
    paths: Dict[int, List[Position]],
    starts: Optional[Dict[int, Position]] = None,
    goals: Optional[Dict[int, Position]] = None,
    figsize: Tuple[int, int] = (6, 6),
) -> None:
    """Visualize a solved MAPF instance.

    This calls `GridMap.draw(...)` which itself uses matplotlib. If matplotlib is
    not available, a warning is printed.
    """
    try:
        grid_map.draw(starts=starts, goals=goals, paths=paths, figsize=figsize)
    except Exception as e:
        warnings.warn(f"Visualization failed: {e}")

