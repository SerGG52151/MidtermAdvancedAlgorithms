# Multi-Agent Pathfinding: Informed Search vs. MARL Performance Evaluation

This project presents a comparative evaluation of two distinct approaches to solving the Multi-Agent Pathfinding (MAPF) problem on a toroidal grid: **Conflict-Based Search (CBS)** and **Multi-Agent Reinforcement Learning (MARL)**.

## 1. Context and Motivation

Pathfinding for multiple agents in shared environments is a fundamental challenge in robotics, automated warehousing, and traffic management. As the number of agents increases, the state space grows exponentially, making the search for an optimal solution computationally expensive (NP-Hard).

In logistics and robotics, autonomous agents must often navi-
gate shared spaces to visit sequences of locations. Traditional
algorithms like A* scale poorly in the joint state space, requir-
ing decoupled approaches like CBS or learning-based methods
like MARL.

## 2. Research Hypothesis

**Guiding Question:** To what extent can a decentralized Multi-Agent Reinforcement Learning (MARL) approach approximate the solution quality (makespan) and execution time of an optimal Conflict-Based Search (CBS) solver?

**Hypothesis:** While CBS will reliably provide an optimal path with no collisions, its computation time will degrade significantly with complexity. Conversely, a trained MARL agent will exhibit near-instantaneous execution times (O(1) inference) but may produce suboptimal paths or fail to converge in highly constrained scenarios.

## 3. Algorithmic Justification

### Approach A: Conflict-Based Search (CBS)
*   **Justification:** CBS is the an algorithm for optimal MAPF. It decomposes the problem into a high-level search (resolving conflicts between agents) and a low-level search (planning individual paths).
*   **Mechanism:** It guarantees optimality by exploring a constraint tree. If Agent A and Agent B collide at time *t*, the search splits: one branch forbids Agent A from being there, the other forbids Agent B.

### Approach B: Multi-Agent Reinforcement Learning (MARL)
*   **Justification:** Learning-based methods shift the computational burden from "execution time" to "training time." Once trained, an agent acts based on a policy function, requiring no search tree exploration during runtime.
*   **Mechanism:** We utilize **Proximal Policy Optimization (PPO)** with a centralized critic and decentralized actors. Agents observe their local surroundings (relative goal position, nearby obstacles) and learn a policy $\pi(a|s)$ to maximize a reward signal based on reaching goals and avoiding collisions.

## 4. System Architecture

The project is structured into modular components separating the environment, the solvers, and the evaluation logic.

*   **`Grid.py`**: The core environment model. It handles the grid state, obstacle definitions, and the toroidal wrapping logic. 
*   **`AStar.py`**: Implements the Classical Solver.
    *   Contains the Low-Level Solver (Space-Time A*).
    *   Contains the High-Level Solver (CBS Constraint Tree).
*   **`train_marl.py`**: The training pipeline for the Learning Solver.
    *   Sets up the PPO model using `stable-baselines3`.
    *   Manages the training loop and model saving.
*   **`gym_wrapper.py`**: The bridge between the `Grid` environment and the RL agent.
    *   Converts grid states into tensor observations (relative coordinates).
    *   Defines the reward function (penalties for collisions, rewards for goals).
*   **`benchmark_comparison.py`**: The evaluation engine.
    *   Runs both solvers on the exact same scenario.
    *   Visualizes paths side-by-side.
    *   Generates comparative metrics (Makespan vs. Execution Time).

## 5. Data Structures and Algorithms

### Implemented Data Structures
1.  **Priority Queue (Min-Heap):** Used in `AStar.py` for the Open Set to efficiently retrieve the node with the lowest f-score.
2.  **Constraint Tree:** A binary tree used in CBS where each node represents a set of constraints imposed on agents to resolve specific collisions.
3.  **Spatio-Temporal Grid:** The A* search operates not just on $(x, y)$ but on $(x, y, t)$, treating time as a third dimension to avoid dynamic obstacles.
4.  **Observation Vector:** In MARL, the grid is flattened into a vector representing relative distances $(dx, dy)$ to goals and neighbors, normalized for neural network input.

### Algorithms
1.  **A* Search (Toroidal):** Modified to handle "wrapping" edges (e.g., moving right from $x=7$ leads to $x=0$).
2.  **Conflict Detection:** An algorithm that iterates through generated paths to find vertex conflicts ($A$ and $B$ are at $v$ at time $t$) and edge conflicts ($A$ and $B$ swap vertices).
3.  **Proximal Policy Optimization (PPO):** A policy gradient method that optimizes the agents' neural network weights by clipping updates to ensure stable learning.

## 6. Requirements & Usage

### Prerequisites
*   Python 3.8+
*   `numpy`, `matplotlib`
*   `gymnasium`, `stable-baselines3`, `shimmy`

### Installation
```bash
pip install numpy matplotlib gymnasium stable-baselines3 shimmy
```

### Running the Comparison
To perform the full evaluation (Training -> Solving -> Comparing):

1.  **Train the MARL Agent:**
    ```bash
    python train_marl.py
    ```
    *(This saves the trained model to `ppo_marl_agent.zip`)*

2.  **Run the Benchmark:**
    ```bash
    python benchmark_comparison.py
    ```
    *(This generates the comparison plot and visualizes both solutions)*
