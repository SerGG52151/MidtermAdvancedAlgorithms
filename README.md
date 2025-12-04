# Multi-Agent Pathfinding on Toroidal Grid

This project implements a Conflict-Based Search (CBS) algorithm combined with A* search to solve Multi-Agent Pathfinding (MAPF) problems on a toroidal grid. Agents are tasked with visiting multiple service locations in a specific sequence and returning to their start position after each service, all while avoiding collisions with other agents.

## Project Structure

- **`AStar.py`**: The core implementation of the pathfinding logic. It includes:
    - `a_star_search`: A* search algorithm adapted for toroidal grids and dynamic constraints.
    - `find_multi_goal_paths`: The main CBS loop that resolves conflicts between agents.
    - `detect_conflicts`: Identifies vertex and edge conflicts between agent paths.
    - `Constraint` and `Conflict` data structures.

- **`AStar_visualize.py`**: Handles the visualization of the results using `matplotlib`.
    - `visualize_paths`: Draws the grid, agents, goals, and paths. It specifically handles the visual representation of paths wrapping around the toroidal grid edges.

- **`benchmark.py`**: An example script that sets up a specific scenario ("Three Services Problem") and runs the solver.
    - Defines the grid size, obstacles (if any), agent start positions, and sequences of goals.
    - Runs the `find_multi_goal_paths` function.
    - Prints performance metrics and path details.
    - Visualizes the solution.

- **`Grid.py`**: Contains helper classes and data structures for grid management, environment simulation, and alternative definitions for positions and constraints. (Note: `AStar.py` currently uses its own internal definitions for some of these, but this file provides a structured environment model).

## Features

- **Toroidal Grid**: The grid wraps around at the edges (top connects to bottom, left connects to right).
- **Multi-Goal Support**: Agents can be assigned a list of goals to visit in order. The current implementation models this as a series of round trips (Start -> Goal -> Start).
- **Conflict-Based Search (CBS)**: A two-level algorithm that finds optimal collision-free paths.
- **Visualization**: Visual output of the paths, including indicators for start positions, service locations, and wrapped path segments.

## Requirements

To run this project, you need Python installed along with the following libraries:

- `numpy`
- `matplotlib`

You can install them using pip:

```bash
pip install numpy matplotlib
```

## Usage

To run the example scenario defined in `benchmark.py`:

```bash
python benchmark.py
```

This will:
1.  Initialize the grid and agent tasks.
2.  Run the CBS solver to find collision-free paths.
3.  Print the paths and statistics to the console.
4.  Open a window showing the visualization of the agents' movements.

## How it Works

1.  **Low-Level Search (A*)**: Finds the optimal path for a single agent respecting a set of spatiotemporal constraints (e.g., "Agent 1 cannot be at (x,y) at time t").
2.  **High-Level Search (CBS)**:
    - Checks the paths found by the low-level search for conflicts (collisions).
    - If a conflict is found (e.g., two agents at the same location at the same time), it splits the search into two branches.
    - In each branch, a constraint is added to prevent one of the agents from using that location at that time, and the low-level search is re-run for that agent.
    - This process continues until a set of conflict-free paths is found.
