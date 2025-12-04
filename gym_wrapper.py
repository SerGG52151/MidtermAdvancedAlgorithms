import gymnasium as gym
import numpy as np
from gymnasium import spaces
from Grid import MAPFEnvironment, Position

class SingleAgentWrapper(gym.Env):
    """
    Envuelve el entorno MAPFEnvironment para hacerlo compatible con Gymnasium.
    Se enfoca en entrenar un solo agente (Agent 0) para navegación básica.
    """
    def __init__(self, env: MAPFEnvironment):
        super().__init__()
        self.env = env
        self.agent_id = 0 
        
        # DEFINIR EL ESPACIO DE ACCIÓN
        # Tu Grid.py tiene 5 acciones: 0=Wait, 1=Up, 2=Right, 3=Down, 4=Left
        self.action_space = spaces.Discrete(5)

        # DEFINIR EL ESPACIO DE OBSERVACIÓN (Lo que ve la IA)
        # Representaremos el mapa como una matriz de números:
        # 0 = Vacío, 1 = Obstáculo, 2 = Agente, 3 = Meta
        self.H = env.grid_map.height
        self.W = env.grid_map.width
        self.observation_space = spaces.Box(
            low=0, 
            high=3, 
            shape=(self.H, self.W), 
            dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        return self._get_observation(), {}

    def step(self, action):
        # 1. Traducir acción de Gym (int) a formato Grid.py (dict)
        joint_action = {self.agent_id: action}
        
        # 2. Ejecutar paso en tu entorno original
        # Guardamos la posición anterior para calcular distancia
        prev_pos = self.env.agent_positions[self.agent_id]
        
        _, done, info = self.env.step(joint_action)
        
        current_pos = self.env.agent_positions[self.agent_id]
        goal_pos = self.env.goals[self.agent_id]

        # 3. SISTEMA DE RECOMPENSAS (CRÍTICO PARA QUE APRENDA)
        reward = -1.0  # Castigo por cada paso (incentiva rapidez)
        
        # Recompensa por llegar a la meta
        if current_pos == goal_pos:
            reward += 50.0
            done = True # Terminar episodio
        
        # Castigo si intenta moverse contra una pared (se queda en el mismo lugar y no era wait)
        # Nota: action 0 es wait. Si action != 0 y no se movió, chocó.
        if action != 0 and prev_pos == current_pos:
            reward -= 2.0 

        # 4. Comprobar truncamiento (límite de tiempo)
        truncated = info.get("time_limit_reached", False)

        return self._get_observation(), reward, done, truncated, info

    def _get_observation(self):
        # Crear una matriz vacía
        obs = np.zeros((self.H, self.W), dtype=np.uint8)
        
        # 1. Dibujar obstáculos
        for obs_pos in self.env.grid_map.obstacles:
            obs[obs_pos.row, obs_pos.col] = 1
            
        # 2. Dibujar al agente
        agent_pos = self.env.agent_positions[self.agent_id]
        obs[agent_pos.row, agent_pos.col] = 2
        
        # 3. Dibujar la meta
        goal_pos = self.env.goals[self.agent_id]
        obs[goal_pos.row, goal_pos.col] = 3
        
        return obs