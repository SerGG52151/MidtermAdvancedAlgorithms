from __future__ import annotations
"""CBS with A* for Multi-Agent Pathfinding on toroidal grids."""

import heapq
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Constraint:
    agent_id: int
    time: int
    position: Optional[Tuple[int, int]] = None
    from_position: Optional[Tuple[int, int]] = None
    to_position: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class Conflict:
    time: int
    agent1: int
    agent2: int
    position: Tuple[int, int]
    conflict_type: str
    position2: Optional[Tuple[int, int]] = None


@dataclass(order=True)
class CTNode:
    num_conflicts: int = field(compare=True)
    cost: int = field(compare=True)
    makespan: int = field(compare=False)
    constraints: Set[Constraint] = field(default_factory=set, compare=False)
    solution: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict, compare=False)
    conflicts: List[Conflict] = field(default_factory=list, compare=False)


def toroidal_distance(pos1: Tuple[int, int], pos2: Tuple[int, int], grid: List[List[int]]) -> int:
    """Manhattan distance on toroidal grid."""
    rows, cols = len(grid), len(grid[0])
    dx, dy = abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])
    return min(dx, rows - dx) + min(dy, cols - dy)


def get_toroidal_neighbors(pos: Tuple[int, int], grid: List[List[int]]) -> List[Tuple[int, int]]:
    """Get valid neighbors on toroidal grid."""
    rows, cols = len(grid), len(grid[0])
    r, c = pos
    neighbors = [((r + dr) % rows, (c + dc) % cols) 
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0), (0,0)]]
    return [n for n in neighbors if grid[n[0]][n[1]] == 0]


def violates_constraint(agent_id: int, from_pos: Tuple[int, int], to_pos: Tuple[int, int],
                       time: int, constraints: Set[Constraint]) -> bool:
    """Check if move violates any constraint."""
    for c in constraints:
        if c.agent_id != agent_id or c.time != time:
            continue
        if c.position and c.position == to_pos:
            return True
        if c.from_position and c.to_position and c.from_position == from_pos and c.to_position == to_pos:
            return True
    return False


def _astar_single_goal(
    start_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    start_time: int,
    grid: List[List[int]],
    constraints: Set[Constraint],
    agent_id: int,
    max_time: int = 500
) -> Optional[List[Tuple[int, int]]]:
    """A* search from start to goal with constraints."""
    open_set = [(toroidal_distance(start_pos, goal_pos, grid), 0, start_time, start_pos, [start_pos])]
    closed_set = set()
    best_g = {(start_time, start_pos): 0}
    counter = 1
    
    while open_set:
        f, _, time, pos, path = heapq.heappop(open_set)
        
        if time > max_time or (time, pos) in closed_set:
            continue
        closed_set.add((time, pos))
        
        if pos == goal_pos:
            return path
        
        for next_pos in get_toroidal_neighbors(pos, grid):
            next_time = time + 1
            if violates_constraint(agent_id, pos, next_pos, next_time, constraints):
                continue
            
            tentative_g = time - start_time + 1
            if (next_time, next_pos) in best_g and best_g[(next_time, next_pos)] <= tentative_g:
                continue
            
            best_g[(next_time, next_pos)] = tentative_g
            f_score = tentative_g + toroidal_distance(next_pos, goal_pos, grid)
            heapq.heappush(open_set, (f_score, counter, next_time, next_pos, path + [next_pos]))
            counter += 1
    
    return None



def a_star_search(
    start: Tuple[int, int],
    goals: List[Tuple[int, int]],  # Lista de waypoints en orden
    grid: List[List[int]],
    constraints: Set[Constraint],
    agent_id: int,
    max_time: int = 1000  # Aumentado para mapas grandes
) -> Optional[List[Tuple[int, int]]]:
    """
    Busca camino secuencial a través de múltiples waypoints.
    No regresa al inicio entre waypoints.
    """
    if not goals:
        return [start]
    
    complete_path = []
    current_pos = start
    current_time = 0
    
    for goal in goals:
        # Buscar camino al siguiente waypoint
        segment = _astar_single_goal(
            current_pos, goal, current_time, 
            grid, constraints, agent_id, max_time
        )
        
        if segment is None:
            return None
        
        # Concatenar, evitando duplicar posición inicial
        if complete_path:
            complete_path.extend(segment[1:])  # Excluir primera posición (ya está)
        else:
            complete_path.extend(segment)
        
        # Actualizar para siguiente waypoint
        current_pos = goal
        current_time = len(complete_path) - 1
    
    return complete_path


def detect_conflicts(paths: Dict[int, List[Tuple[int, int]]]) -> List[Conflict]:
    """Detect all conflicts between agent paths."""
    if not paths:
        return []
    
    max_time = max(len(path) for path in paths.values())
    agent_ids = list(paths.keys())
    get_pos = lambda aid, t: paths[aid][t] if t < len(paths[aid]) else paths[aid][-1]
    conflicts = []
    
    for t in range(max_time):
        # Vertex conflicts
        positions = {}
        for aid in agent_ids:
            pos = get_pos(aid, t)
            if pos in positions:
                conflicts.append(Conflict(t, positions[pos], aid, pos, 'vertex'))
            else:
                positions[pos] = aid
        
        # Edge conflicts
        if t < max_time - 1:
            for i in range(len(agent_ids)):
                for j in range(i + 1, len(agent_ids)):
                    a1, a2 = agent_ids[i], agent_ids[j]
                    p1_t, p1_t1 = get_pos(a1, t), get_pos(a1, t + 1)
                    p2_t, p2_t1 = get_pos(a2, t), get_pos(a2, t + 1)
                    if p1_t == p2_t1 and p2_t == p1_t1 and p1_t != p1_t1:
                        conflicts.append(Conflict(t + 1, a1, a2, p1_t, 'edge', p1_t1))
    
    return conflicts


def create_constraints_from_conflict(conflict: Conflict) -> List[Set[Constraint]]:
    """Create constraint sets for resolving a conflict."""
    if conflict.conflict_type == 'vertex':
        return [
            {Constraint(conflict.agent1, conflict.time, position=conflict.position)},
            {Constraint(conflict.agent2, conflict.time, position=conflict.position)}
        ]
    else:
        return [
            {Constraint(conflict.agent1, conflict.time, from_position=conflict.position, to_position=conflict.position2)},
            {Constraint(conflict.agent2, conflict.time, from_position=conflict.position2, to_position=conflict.position)}
        ]


def find_multi_goal_paths(
    starts: List[Tuple[int, int]],
    goals_list: List[List[Tuple[int, int]]],
    grid: List[List[int]],
    max_iterations: int = 100000,
    verbose: bool = False
) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    """
    Conflict-Based Search (CBS) for multi-goal pathfinding on a toroidal grid.
    Each agent makes round trips: home -> service -> home for each service.
    """
    num_agents = len(starts)
    root_constraints = set()
    root_solution = {}
    
    if verbose:
        print("Finding initial paths...")
    
    for agent_id in range(num_agents):
        path = a_star_search(starts[agent_id], goals_list[agent_id], grid, root_constraints, agent_id)
        if path is None:
            if verbose:
                print(f"No initial path found for agent {agent_id}")
            return None
        root_solution[agent_id] = path
        if verbose:
            print(f"  Agent {agent_id}: {len(path)} steps")
    
    root_conflicts = detect_conflicts(root_solution)
    root_cost = sum(len(path) for path in root_solution.values())
    root_makespan = max(len(path) for path in root_solution.values())
    
    if verbose:
        print(f"Initial solution has {len(root_conflicts)} conflicts, cost={root_cost}, makespan={root_makespan}")
    
    root_node = CTNode(
        num_conflicts=len(root_conflicts),
        cost=root_cost,
        makespan=root_makespan,
        constraints=root_constraints,
        solution=root_solution,
        conflicts=root_conflicts
    )
    
    if not root_conflicts:
        if verbose:
            print("No conflicts in initial solution!")
        return root_solution
    
    open_list = [root_node]
    iterations = 0
    best_so_far = len(root_conflicts)
    
    while open_list and iterations < max_iterations:
        iterations += 1
        current_node = heapq.heappop(open_list)
        
        if verbose and iterations % 5000 == 0:
            print(f"Iteration {iterations}, conflicts: {len(current_node.conflicts)}, cost: {current_node.cost}")
        
        if len(current_node.conflicts) < best_so_far:
            best_so_far = len(current_node.conflicts)
            if verbose:
                print(f"  New best: {best_so_far} conflicts at iteration {iterations}, cost={current_node.cost}")
        
        if not current_node.conflicts:
            if verbose:
                print(f"CBS found solution in {iterations} iterations")
            return current_node.solution
        
        conflict = current_node.conflicts[0]
        constraint_sets = create_constraints_from_conflict(conflict)
        
        for new_constraint_set in constraint_sets:
            child_constraints = current_node.constraints | new_constraint_set
            affected_agent = list(new_constraint_set)[0].agent_id
            
            new_path = a_star_search(starts[affected_agent], goals_list[affected_agent], 
                                    grid, child_constraints, affected_agent, max_time=3000)
            
            if new_path is None:
                continue
            
            child_solution = dict(current_node.solution)
            child_solution[affected_agent] = new_path
            child_conflicts = detect_conflicts(child_solution)
            child_cost = sum(len(path) for path in child_solution.values())
            child_makespan = max(len(path) for path in child_solution.values())
            
            child_node = CTNode(
                num_conflicts=len(child_conflicts),
                cost=child_cost,
                makespan=child_makespan,
                constraints=child_constraints,
                solution=child_solution,
                conflicts=child_conflicts
            )
            
            heapq.heappush(open_list, child_node)
    
    if verbose:
        print(f"CBS failed to find solution after {iterations} iterations")
        print(f"Best solution had {best_so_far} conflicts")
    return None
