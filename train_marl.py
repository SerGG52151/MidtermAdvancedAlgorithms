"""
Entrenamiento MARL simple que resuelve el MISMO problema que CBS.
"""

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from mission_env import generate_large_map, generate_missions, MissionMAPFEnvironment
from gym_wrapper import MissionMAPFVecEnv

MAP_WIDTH = 8
MAP_HEIGHT = 8
NUM_AGENTS = 2

def create_env_difficulty(difficulty_level):
    """
    Nivel 1: Metas a 1-2 pasos (Aprender a moverse y no chocar).
    Nivel 2: Metas a 3-5 pasos (Navegación media).
    Nivel 3: Mapa completo (Navegación compleja).
    """
    def _init():
        grid = generate_large_map(width=MAP_WIDTH, height=MAP_HEIGHT, obstacle_density=0.1, seed=42)
        
        max_dist = None
        if difficulty_level == 1: max_dist = 2
        elif difficulty_level == 2: max_dist = 5
        
        missions = generate_missions(
            grid, 
            num_agents=NUM_AGENTS, 
            num_waypoints=1, 
            min_distance=1 if max_dist is None else max_dist,
            seed=None # Semilla aleatoria para variar misiones cada reset
        )
        return MissionMAPFEnvironment(grid, missions, max_time=50)
    return _init

def main():
    print(" Iniciando entrenamiento MARL")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- NIVEL 1: KINDERGARTEN (Distancia máx 2) ---
    print("\n NIVEL 1: Aprender a dar el paso final (Distancia 1-2)")
    env_lvl1 = MissionMAPFVecEnv(create_env_difficulty(1), num_agents=NUM_AGENTS)
    
    model = PPO(
        "MultiInputPolicy",
        env_lvl1,
        verbose=1,
        device=device,
        learning_rate=5e-4, # Tasa alta para aprender rápido lo básico
        n_steps=1024,
        batch_size=64,
        ent_coef=0.05,      # Exploración moderada
    )
    model.learn(total_timesteps=30_000)
    model.save("ppo_lvl1")
    env_lvl1.close()
    
    # --- NIVEL 2: ESCUELA (Distancia máx 5) ---
    print("\n📚 NIVEL 2: Navegación media (Distancia 3-5)")
    # Cargamos el cerebro del nivel 1 para seguir aprendiendo
    env_lvl2 = MissionMAPFVecEnv(create_env_difficulty(2), num_agents=NUM_AGENTS)
    model.set_env(env_lvl2) # Cambiamos al entorno más difícil
    
    model.learn(total_timesteps=50_000)
    model.save("ppo_lvl2")
    env_lvl2.close()
    
    # --- NIVEL 3: UNIVERSIDAD (Mapa completo) ---
    print("\n📚 NIVEL 3: Mapa Completo (Sin límites)")
    env_lvl3 = MissionMAPFVecEnv(create_env_difficulty(3), num_agents=NUM_AGENTS)
    model.set_env(env_lvl3)
    
    # Bajamos el learning rate para refinar detalles
    model.learning_rate = 3e-4
    model.learn(total_timesteps=100_000)
    
    print("\n ENTRENAMIENTO FINALIZADO")
    model.save("ppo_simple_marl")

if __name__ == "__main__":
    main()