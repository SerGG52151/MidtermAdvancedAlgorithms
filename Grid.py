from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Optional, Set
import warnings

try:
    import matplotlib.pyplot as plt
    from matplotlib import colors
except Exception:  # pragma: no cover - optional plotting
    plt = None
    colors = None


@dataclass(frozen=True)
class Position:
    row: int
    col: int


@dataclass(frozen=True)
class Constraint:
    agent_id: int
    time: int
    position: Optional[Position] = None
    from_position: Optional[Position] = None
    to_position: Optional[Position] = None


@dataclass(frozen=True)
class Conflict:
    time: int
    agents: Tuple[int, int]
    kind: str
    pos1: Position
    pos2: Position


class GridMap:
    def __init__(
        self,
        width: int,
        height: int,
        obstacles: Optional[Iterable[Position]] = None,
        allow_diagonal: bool = False,
        allow_wait: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.obstacles: Set[Position] = set(obstacles or [])
        self.allow_diagonal = allow_diagonal
        self.allow_wait = allow_wait
        self.adjacency_list: Dict[Position, List[Position]] = {}
        self._build_adjacency_list()

    @classmethod
    def from_ascii(cls, ascii_map: List[str], **kwargs) -> "GridMap":
        height = len(ascii_map)
        if height == 0:
            raise ValueError("ascii_map must contain at least one row")
        width = len(ascii_map[0])

        obstacles: List[Position] = []
        for r, row in enumerate(ascii_map):
            if len(row) != width:
                raise ValueError("All rows in ascii_map must have the same length")
            for c, ch in enumerate(row):
                if ch == "#":
                    obstacles.append(Position(r, c))

        return cls(width=width, height=height, obstacles=obstacles, **kwargs)

    def _build_adjacency_list(self) -> None:
        if self.allow_diagonal:
            directions = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
            ]
        else:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for r in range(self.height):
            for c in range(self.width):
                pos = Position(r, c)
                if pos in self.obstacles:
                    continue
                neighbors: List[Position] = []
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    npos = Position(nr, nc)
                    if self.in_bounds(npos) and npos not in self.obstacles:
                        neighbors.append(npos)
                if self.allow_wait:
                    neighbors.append(pos)
                self.adjacency_list[pos] = neighbors

    def draw(
        self,
        starts: Optional[Dict[int, Position]] = None,
        goals: Optional[Dict[int, Position]] = None,
        paths: Optional[Dict[int, List[Position]]] = None,
        figsize: Tuple[int, int] = (6, 6),
    ) -> None:
        """Draw the grid, obstacles, starts/goals and optional paths using matplotlib.

        - obstacles are black squares
        - free cells are white
        - starts marked with blue circles, goals with green stars
        - paths drawn with colored lines per agent and annotated with agent id
        """
        if plt is None or colors is None:
            warnings.warn("matplotlib is not available. Install matplotlib to enable drawing.")
            return

        grid_img = [[0 if Position(r, c) not in self.obstacles else 1 for c in range(self.width)] for r in range(self.height)]

        cmap = colors.ListedColormap(["white", "black"])
        bounds = [0, 0.5, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(grid_img, cmap=cmap, norm=norm)

        # draw grid lines
        ax.set_xticks([x - 0.5 for x in range(1, self.width)], minor=False)
        ax.set_yticks([y - 0.5 for y in range(1, self.height)], minor=False)
        ax.grid(which="major", color="gray", linestyle="-", linewidth=0.5)

        # optional: draw starts and goals
        if starts:
            for aid, p in starts.items():
                ax.plot(p.col, p.row, marker="o", color="blue", markersize=10)
                ax.text(p.col + 0.1, p.row + 0.1, str(aid), color="white", fontsize=8)

        if goals:
            for aid, p in goals.items():
                ax.plot(p.col, p.row, marker="*", color="green", markersize=12)

        # draw paths
        if paths:
            import random

            for aid, path in paths.items():
                cols = [p.col for p in path]
                rows = [p.row for p in path]
                color = f"C{aid % 10}"
                ax.plot(cols, rows, marker="o", color=color, linewidth=2, markersize=4)
                # annotate path with agent id at start
                if path:
                    ax.text(cols[0] + 0.1, rows[0] + 0.1, f"A{aid}", color=color, fontsize=8, weight="bold")

        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        # Save a copy to the working directory for headless inspection
        import os

        out_path = os.path.join(os.getcwd(), "benchmark_plot.png")
        try:
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            print(f"Saved plot to {out_path}")
        except Exception as e:
            warnings.warn(f"Failed to save plot to {out_path}: {e}")
        plt.show()

    def in_bounds(self, pos: Position) -> bool:
        return 0 <= pos.row < self.height and 0 <= pos.col < self.width

    def is_free(self, pos: Position) -> bool:
        return self.in_bounds(pos) and pos not in self.obstacles

    def neighbors(self, pos: Position) -> List[Position]:
        return self.adjacency_list.get(pos, [])


class ReservationTable:
    def __init__(self) -> None:
        self.vertex_reservations: Dict[int, Set[Position]] = defaultdict(set)
        self.edge_reservations: Dict[int, Set[Tuple[Position, Position]]] = defaultdict(set)

    def reserve_path(self, path: List[Position]) -> None:
        if not path:
            return
        prev = path[0]
        self.vertex_reservations[0].add(prev)
        for t in range(1, len(path)):
            curr = path[t]
            self.vertex_reservations[t].add(curr)
            self.edge_reservations[t].add((prev, curr))
            prev = curr

    def is_vertex_conflict(self, pos: Position, time: int) -> bool:
        return pos in self.vertex_reservations.get(time, set())

    def is_edge_conflict(self, from_pos: Position, to_pos: Position, time: int) -> bool:
        edges = self.edge_reservations.get(time, set())
        return (from_pos, to_pos) in edges or (to_pos, from_pos) in edges

    def clear(self) -> None:
        self.vertex_reservations.clear()
        self.edge_reservations.clear()


def violates_constraints(
    agent_id: int,
    from_pos: Position,
    to_pos: Position,
    time: int,
    constraints: Optional[Iterable[Constraint]],
) -> bool:
    if not constraints:
        return False

    for c in constraints:
        if c.agent_id != agent_id or c.time != time:
            continue
        if c.position is not None and c.position == to_pos:
            return True
        if (
            c.from_position is not None
            and c.to_position is not None
            and c.from_position == from_pos
            and c.to_position == to_pos
        ):
            return True

    return False


class MAPFEnvironment:
    ACTIONS: Dict[int, Tuple[int, int]] = {
        0: (0, 0),
        1: (-1, 0),
        2: (0, 1),
        3: (1, 0),
        4: (0, -1),
    }

    def __init__(
        self,
        grid_map: GridMap,
        starts: Dict[int, Position],
        goals_list: Dict[int, List[Position]],
        max_time: Optional[int] = None,
    ) -> None:
        self.grid_map = grid_map
        self.starts = dict(starts)
        self.goals_list = dict(goals_list)
        self.max_time = max_time
        self.time: int = 0
        self.agent_positions: Dict[int, Position] = dict(starts)
        
        # Track current goal index for each agent
        self.agent_goal_indices: Dict[int, int] = {aid: 0 for aid in starts}
        # Track if agent has finished all goals
        self.agent_finished: Dict[int, bool] = {aid: False for aid in starts}
        # Current active goal for each agent
        self.current_goals: Dict[int, Position] = {}
        self._update_current_goals()

    def _update_current_goals(self):
        for aid in self.starts:
            if self.agent_finished[aid]:
                self.current_goals[aid] = self.agent_positions[aid] # Stay put
                continue
                
            goals = self.goals_list[aid]
            idx = self.agent_goal_indices[aid]
            
            if idx < len(goals):
                self.current_goals[aid] = goals[idx]
            else:
                self.agent_finished[aid] = True
                self.current_goals[aid] = self.agent_positions[aid]

    def reset(self) -> Dict[int, Position]:
        self.time = 0
        self.agent_positions = dict(self.starts)
        self.agent_goal_indices = {aid: 0 for aid in self.starts}
        self.agent_finished = {aid: False for aid in self.starts}
        self._update_current_goals()
        return dict(self.agent_positions)

    def successors_for_agent(
        self,
        agent_id: int,
        pos: Position,
        time: int,
        constraints: Optional[Iterable[Constraint]] = None,
        reservation_table: Optional[ReservationTable] = None,
    ) -> List[Position]:
        successors: List[Position] = []
        for nxt in self.grid_map.neighbors(pos):
            t_next = time + 1
            if violates_constraints(agent_id, pos, nxt, t_next, constraints):
                continue
            if reservation_table is not None:
                if reservation_table.is_vertex_conflict(nxt, t_next):
                    continue
                if reservation_table.is_edge_conflict(pos, nxt, t_next):
                    continue
            successors.append(nxt)
        return successors

    def step(
        self,
        joint_action: Dict[int, int],
    ) -> Tuple[Dict[int, Position], bool, Dict]:
        current_positions = dict(self.agent_positions)
        proposed_positions: Dict[int, Position] = {}
        moves: Dict[int, Tuple[Position, Position]] = {}

        for agent_id, pos in current_positions.items():
            action = joint_action.get(agent_id, 0)
            if action not in self.ACTIONS:
                raise ValueError(f"Invalid action {action} for agent {agent_id}")
            dr, dc = self.ACTIONS[action]
            candidate = Position(pos.row + dr, pos.col + dc)
            if not self.grid_map.is_free(candidate):
                candidate = pos
            proposed_positions[agent_id] = candidate
            moves[agent_id] = (pos, candidate)

        vertex_conflicts: List[Conflict] = []
        edge_conflicts: List[Conflict] = []

        cell_to_agents: Dict[Position, List[int]] = defaultdict(list)
        for agent_id, new_pos in proposed_positions.items():
            cell_to_agents[new_pos].append(agent_id)

        for pos, agents in cell_to_agents.items():
            if len(agents) > 1:
                for i in range(len(agents)):
                    for j in range(i + 1, len(agents)):
                        vertex_conflicts.append(
                            Conflict(
                                time=self.time + 1,
                                agents=(agents[i], agents[j]),
                                kind="vertex",
                                pos1=pos,
                                pos2=pos,
                            )
                        )

        agent_ids = list(current_positions.keys())
        for i in range(len(agent_ids)):
            a_i = agent_ids[i]
            from_i, to_i = moves[a_i]
            for j in range(i + 1, len(agent_ids)):
                a_j = agent_ids[j]
                from_j, to_j = moves[a_j]
                if from_i == to_j and from_j == to_i and to_i != to_j:
                    edge_conflicts.append(
                        Conflict(
                            time=self.time + 1,
                            agents=(a_i, a_j),
                            kind="edge",
                            pos1=from_i,
                            pos2=to_i,
                        )
                    )

        self.agent_positions = proposed_positions
        self.time += 1

        # Check for goal completion
        for aid, pos in self.agent_positions.items():
            if not self.agent_finished[aid]:
                if pos == self.current_goals[aid]:
                    self.agent_goal_indices[aid] += 1
                    self._update_current_goals()

        all_at_goal = all(self.agent_finished.values())
        time_limit_reached = self.max_time is not None and self.time >= self.max_time
        done = all_at_goal or time_limit_reached

        info = {
            "time": self.time,
            "vertex_conflicts": vertex_conflicts,
            "edge_conflicts": edge_conflicts,
            "all_at_goal": all_at_goal,
            "time_limit_reached": time_limit_reached,
            "agent_finished": self.agent_finished
        }

        return dict(self.agent_positions), done, info


    def render(self, starts: Optional[Dict[int, Position]] = None, goals: Optional[Dict[int, Position]] = None, paths: Optional[Dict[int, List[Position]]] = None, figsize: Tuple[int, int] = (6, 6)) -> None:
        """Convenience render that delegates to GridMap.draw.

        This lets users call `env.render(...)` to visualize the current map and optional paths.
        """
        self.grid_map.draw(starts=starts, goals=goals, paths=paths, figsize=figsize)


def detect_conflicts(
    paths: Dict[int, List[Position]],
) -> List[Conflict]:
    if not paths:
        return []

    conflicts: List[Conflict] = []
    max_T = max(len(p) for p in paths.values())
    agent_ids = list(paths.keys())

    def get_pos(agent_id: int, t: int) -> Position:
        path = paths[agent_id]
        if t < len(path):
            return path[t]
        return path[-1]

    for t in range(max_T):
        cell_to_agents: Dict[Position, List[int]] = defaultdict(list)
        for agent_id in agent_ids:
            pos = get_pos(agent_id, t)
            cell_to_agents[pos].append(agent_id)

        for pos, agents in cell_to_agents.items():
            if len(agents) > 1:
                for i in range(len(agents)):
                    for j in range(i + 1, len(agents)):
                        conflicts.append(
                            Conflict(
                                time=t,
                                agents=(agents[i], agents[j]),
                                kind="vertex",
                                pos1=pos,
                                pos2=pos,
                            )
                        )

        if t == max_T - 1:
            break

        for i in range(len(agent_ids)):
            a_i = agent_ids[i]
            from_i = get_pos(a_i, t)
            to_i = get_pos(a_i, t + 1)
            for j in range(i + 1, len(agent_ids)):
                a_j = agent_ids[j]
                from_j = get_pos(a_j, t)
                to_j = get_pos(a_j, t + 1)
                if from_i == to_j and from_j == to_i and to_i != to_j:
                    conflicts.append(
                        Conflict(
                            time=t + 1,
                            agents=(a_i, a_j),
                            kind="edge",
                            pos1=from_i,
                            pos2=to_i,
                        )
                    )

    return conflicts

