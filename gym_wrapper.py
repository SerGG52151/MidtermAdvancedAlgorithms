import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from mission_env import MissionMAPFEnvironment, Mission

class MissionMAPFVecEnv(VecEnv):
    def __init__(self, env_builder_fn, num_agents):
        self.mission_env = env_builder_fn()
        self.num_agents = num_agents
        self.agent_ids = list(self.mission_env.missions.keys())
        
        # Acción: 5 acciones discretas
        action_space = spaces.Discrete(5)
        
        self.H = self.mission_env.grid_map.height
        self.W = self.mission_env.grid_map.width
        
        # Usamos Dict para combinar la "Imagen" con el "Vector"
        observation_space = spaces.Dict({
            # La visión del mapa (para ver obstaculos y otros agentes)
            "map": spaces.Box(low=0, high=1, shape=(5, self.H, self.W), dtype=np.float32),
            
            # La "Brújula": Vector (dx, dy) normalizado hacia el objetivo
            # Esto le dice explícitamente hacia dónde ir.
            "vec": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        })
        
        self.num_envs = num_agents
        self.observation_space = observation_space
        self.action_space = action_space
        self.actions_buffer = {}
        self.prev_waypoints = {aid: 0 for aid in self.agent_ids}
    
    def reset(self):
        """Reinicia el entorno y devuelve observaciones iniciales."""
        self.mission_env.reset()
        self.actions_buffer = {}
        self.prev_waypoints = {aid: 0 for aid in self.agent_ids}
        
        obs_dict = self._get_all_observations()
        return obs_dict
    
    def step_async(self, actions):
        """Almacena acciones para ejecución sincronizada."""
        self.actions_buffer = {
            aid: action for aid, action in zip(self.agent_ids, actions)
        }
    
    def step_wait(self):
        """Ejecuta paso sincronizado y calcula recompensas."""
        prev_dists = {aid: self._get_distance_to_target(aid) for aid in self.agent_ids}
        prev_waypoints = self.prev_waypoints.copy()
        
        # Guardamos posiciones ANTES de movernos para detectar choques
        positions_before = self.mission_env.agent_positions.copy()
        
        # Ejecutar simulación
        positions, done_global, info = self.mission_env.step(self.actions_buffer)
        
        obs_list = self._get_all_observations()
        rew_list = []
        dones_list = []
        infos_list = []
        
        conflicts = len(info.get("vertex_conflicts", [])) + len(info.get("edge_conflicts", []))
        
        for i, agent_id in enumerate(self.agent_ids):
            reward = 0.0
            
            # Datos de estado
            curr_dist = self._get_distance_to_target(agent_id)
            progress = prev_dists[agent_id] - curr_dist
            
            # Acción que intentó ejecutar el agente
            action_attempted = self.actions_buffer.get(agent_id, 0)
            
            # --- DETECCIÓN DE CHOQUE CONTRA PARED ---
            # Si intentó moverse (Acción != 0) PERO su posición es la misma
            is_stuck_on_wall = (action_attempted != 0) and \
                               (positions_before[agent_id] == positions[agent_id])
            
            # 1. Recompensa por Progreso (Distance Shaping)
            if progress > 0:
                reward += 0.5 + (progress * 0.1) # Buen trabajo
            elif progress < 0:
                reward -= 0.1 # Te alejaste un poco
            
            # 2. CASTIGO POR CHOCAR (La corrección clave)
            if is_stuck_on_wall:
                reward -= 0.5  # ¡Castigo fuerte por intentar atravesar paredes!
            
            # 3. Penalización por Paso (Time penalty)
            reward -= 0.01 
            
            # 4. Recompensa GIGANTE por Waypoint (Objetivo intermedio)
            mission = self.mission_env.missions[agent_id]
            curr_waypoint = mission.current_waypoint_idx
            
            if curr_waypoint > prev_waypoints[agent_id]:
                reward += 10.0 # ¡Llegó al waypoint!
                
            # 5. Recompensa por Completar Misión Final
            if agent_id in info.get("completed_agents", []):
                reward += 20.0 # ¡Terminó!

            # 6. Penalización por conflictos con otros agentes
            if conflicts > 0:
                reward -= 0.25 * conflicts
            
            rew_list.append(reward)
            dones_list.append(done_global)
            
            # Info para logs
            agent_info = info.copy()
            agent_info["distance_to_target"] = curr_dist
            infos_list.append(agent_info)
        
        # Actualizar historial
        for agent_id in self.agent_ids:
            self.prev_waypoints[agent_id] = self.mission_env.missions[agent_id].current_waypoint_idx
        
        if done_global:
            # Guardar observaciones actuales como terminal_observation ANTES de resetear
            obs_before_reset = obs_list.copy()
            new_obs = self.reset()
            for i in range(len(infos_list)):
                # Crear diccionario con la observación del agente i
                infos_list[i]["terminal_observation"] = {
                    "map": obs_before_reset["map"][i],
                    "vec": obs_before_reset["vec"][i]
                }
            obs_list = new_obs
        
        return obs_list, np.array(rew_list), np.array(dones_list), infos_list
    
    def _get_distance_to_target(self, agent_id: int) -> int:
        """Distancia Manhattan al waypoint actual."""
        agent_pos = self.mission_env.agent_positions[agent_id]
        target_pos = self.mission_env.goals[agent_id]
        return abs(agent_pos.row - target_pos.row) + abs(agent_pos.col - target_pos.col)
    
    def _get_all_observations(self):
        obs_batch = []
        
        # (Lógica de mapas igual que antes...)
        walls = np.zeros((self.H, self.W), dtype=np.float32)
        for obs in self.mission_env.grid_map.obstacles: walls[obs.row, obs.col] = 1.0
        
        all_agents_map = np.zeros((self.H, self.W), dtype=np.float32)
        for aid, pos in self.mission_env.agent_positions.items(): all_agents_map[pos.row, pos.col] = 1.0
        
        for agent_id in self.agent_ids:
            # --- PARTE 1: MAPA (Igual que antes) ---
            target = self.mission_env.goals[agent_id]
            pos = self.mission_env.agent_positions[agent_id]
            
            ch0 = walls.copy()
            ch1 = np.zeros((self.H, self.W), dtype=np.float32); ch1[target.row, target.col] = 1.0
            ch2 = np.zeros((self.H, self.W), dtype=np.float32); ch2[pos.row, pos.col] = 1.0
            ch3 = all_agents_map.copy(); ch3[pos.row, pos.col] = 0.0
            
            # Canal 4 (Densidad) omitido por brevedad, puedes dejarlo si quieres
            # Usamos zeros para rellenar si quitaste densidad o copialo de tu código anterior
            ch4 = np.zeros((self.H, self.W), dtype=np.float32) 
            
            map_obs = np.stack([ch0, ch1, ch2, ch3, ch4])
            
            # --- PARTE 2: EL VECTOR MÁGICO (La Brújula) ---
            # Calculamos dirección normalizada
            dy = (target.row - pos.row) / self.H
            dx = (target.col - pos.col) / self.W
            vec_obs = np.array([dy, dx], dtype=np.float32)
            
            # Empaquetamos en diccionario
            obs_batch.append({
                "map": map_obs,
                "vec": vec_obs
            })
            
        # IMPORTANTE: VecEnv espera un array de diccionarios o un diccionario de arrays
        # SB3 prefiere "Diccionario de arrays stackeados"
        return {
            "map": np.stack([o["map"] for o in obs_batch]),
            "vec": np.stack([o["vec"] for o in obs_batch])
        }
    
    # Métodos requeridos por VecEnv
    def close(self):
        pass
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_agents
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return []
    
    def get_attr(self, attr_name, indices=None):
        return []
    
    def set_attr(self, attr_name, value, indices=None):
        pass
    
    def seed(self, seed=None):
        pass