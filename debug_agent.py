"""
Diagnóstico detallado de qué hacen los agentes entrenados.
"""

import torch
import numpy as np
from stable_baselines3 import PPO
from mission_env import generate_large_map, generate_missions, MissionMAPFEnvironment
from gym_wrapper import MissionMAPFVecEnv

def create_debug_env():
    """Mismo entorno que usaste para entrenar."""
    grid_map = generate_large_map(
        width=8, height=8, obstacle_density=0.15, seed=42, min_free_paths=2
    )
    
    missions = generate_missions(
        grid_map, num_agents=2, num_waypoints=1, min_distance=4, seed=100
    )
    
    return MissionMAPFEnvironment(grid_map, missions, max_time=100)

def analyze_agent_behavior(model_path="ppo_simple_marl"):
    """Analiza en detalle el comportamiento del agente."""
    
    print("=" * 70)
    print("DIAGNÓSTICO DE COMPORTAMIENTO DEL AGENTE")
    print("=" * 70)
    
    # Cargar modelo
    env = MissionMAPFVecEnv(create_debug_env, num_agents=2)
    model = PPO.load(model_path, env=env)
    
    # Ejecutar un episodio con análisis detallado
    obs = env.reset()
    total_rewards = [0, 0]
    action_counts = {0: {}, 1: {}}
    stuck_count = [0, 0]
    
    print("\n Posiciones iniciales de los agentes:")
    for i in range(2):
        agent_pos = env.mission_env.agent_positions[i]
        target_pos = env.mission_env.goals[i]
        dist = abs(agent_pos.row - target_pos.row) + abs(agent_pos.col - target_pos.col)
        print(f"  Agente {i}: Posición {agent_pos}, Objetivo {target_pos}, Distancia {dist}")
    
    print("\n🎮 Comenzando simulación paso a paso:")
    print("-" * 70)
    
    for step in range(30):
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)
        
        # Contar acciones
        for i, action in enumerate(actions):
            action_counts[i][action] = action_counts[i].get(action, 0) + 1
        
        # Acumular recompensas
        total_rewards[0] += rewards[0]
        total_rewards[1] += rewards[1]
        
        # Mostrar información detallada
        print(f"\nPaso {step}:")
        for i in range(2):
            agent_pos = env.mission_env.agent_positions[i]
            target_pos = env.mission_env.goals[i]
            dist = abs(agent_pos.row - target_pos.row) + abs(agent_pos.col - target_pos.col)
            
            # Mapear acción a texto
            action_map = {0: "WAIT", 1: "UP", 2: "RIGHT", 3: "DOWN", 4: "LEFT"}
            action_text = action_map.get(actions[i], f"UNK({actions[i]})")
            
            print(f"  Agente {i}: Acción={action_text}, Pos={agent_pos}, "
                  f"Dist={dist}, Recomp={rewards[i]:.2f}")
            
            # Detectar si está atascado
            if step > 3 and dist == infos[i].get("prev_dist", float('inf')):
                stuck_count[i] += 1
            infos[i]["prev_dist"] = dist
        
        if all(dones):
            print("\n Episodio terminado temprano")
            break
    
    # Análisis final
    print("\n" + "=" * 70)
    print("ANÁLISIS FINAL")
    print("=" * 70)
    
    print("\n Distribución de acciones:")
    for i in range(2):
        print(f"\nAgente {i}:") 
        total_actions = sum(action_counts[i].values())
        for action, count in sorted(action_counts[i].items()):
            percentage = (count / total_actions) * 100
            action_map = {0: "WAIT", 1: "UP", 2: "RIGHT", 3: "DOWN", 4: "LEFT"}
            print(f"  {action_map.get(action, action)}: {count} veces ({percentage:.1f}%)")
    
    print(f"\n Recompensas totales: Agente 0={total_rewards[0]:.2f}, Agente 1={total_rewards[1]:.2f}")
    print(f" Pasos atascados: Agente 0={stuck_count[0]}, Agente 1={stuck_count[1]}")
    
    # Verificar si alcanzaron objetivos
    final_positions = env.mission_env.agent_positions
    final_targets = env.mission_env.goals
    
    print(f"\n Estado final:")
    for i in range(2):
        reached = final_positions[i] == final_targets[i]
        status = " ALCANZÓ" if reached else " NO ALCANZÓ"
        print(f"  Agente {i}: {status} (Pos: {final_positions[i]}, Obj: {final_targets[i]})")
    
    return env.mission_env

def visualize_agent_path(env):
    """Visualiza el camino que tomaron los agentes."""
    print("\n" + "=" * 70)
    print("VISUALIZACIÓN DEL MAPA")
    print("=" * 70)
    
    # Crear una representación ASCII del mapa
    grid = env.grid_map
    width, height = grid.width, grid.height
    
    # Inicializar mapa vacío
    ascii_map = [['.' for _ in range(width)] for _ in range(height)]
    
    # Marcar obstáculos
    for obs in grid.obstacles:
        ascii_map[obs.row][obs.col] = '#'
    
    # Marcar posiciones finales de agentes
    for agent_id, pos in env.agent_positions.items():
        ascii_map[pos.row][pos.col] = str(agent_id)
    
    # Marcar objetivos
    for agent_id, goal in env.goals.items():
        if ascii_map[goal.row][goal.col] == '.':
            ascii_map[goal.row][goal.col] = 'G'
        elif ascii_map[goal.row][goal.col].isdigit():
            # Si un agente está en su objetivo
            ascii_map[goal.row][goal.col] = '★'
    
    print("\nMapa final ('.'=libre, '#'=obstáculo, número=agente, 'G'=objetivo, '★'=agente en objetivo):")
    print("-" * (width * 2 + 1))
    for row in ascii_map:
        print(" " + " ".join(row))
    print("-" * (width * 2 + 1))

def train_with_focused_learning():
    """Entrenamiento enfocado en resolver el problema del 'paso final'."""
    
    print("=" * 70)
    print("ENTRENAMIENTO ENFOCADO EN 'PASO FINAL'")
    print("=" * 70)
    
    # Crear entorno
    env = MissionMAPFVecEnv(create_debug_env, num_agents=2)
    
    # Hiperparámetros optimizados para este problema específico
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=5e-4,  # Más alto para aprender más rápido
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.1,  # Exploración moderada
        vf_coef=0.5,
        max_grad_norm=0.5,
        # Añadir política más grande para problemas complejos
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        )
    )
    
    # Entrenamiento por fases con evaluación
    print("\n FASE 1: Aprendizaje básico de navegación (100k pasos)")
    model.learn(total_timesteps=100_000, reset_num_timesteps=True)
    model.save("ppo_phase1")
    
    # Evaluar
    print("\n Evaluando Fase 1...")
    test_performance(model, env, "Fase 1")
    
    print("\n FASE 2: Enfocado en precisión (100k pasos más)")
    model.learn(total_timesteps=100_000, reset_num_timesteps=False)
    model.save("ppo_phase2")
    
    # Evaluar
    print("\n Evaluando Fase 2...")
    test_performance(model, env, "Fase 2")
    
    print("\n FASE 3: Refinamiento final (50k pasos)")
    model.learn(total_timesteps=50_000, reset_num_timesteps=False)
    model.save("ppo_final_tuned")
    
    print("\n Entrenamiento completado!")
    return model, env

def test_performance(model, env, phase_name):
    """Evalúa el rendimiento del modelo."""
    
    success_count = 0
    total_steps = []
    
    for ep in range(5):
        obs = env.reset()
        steps = 0
        done = False
        
        while not done and steps < 50:
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            done = all(dones)
            steps += 1
        
        # Verificar éxito
        if infos and infos[0].get("missions_complete", False):
            success_count += 1
            print(f"  Episodio {ep+1}:  Completado en {steps} pasos")
        else:
            # Mostrar por qué falló
            for i, info in enumerate(infos[:2]):
                dist = info.get("distance_to_target", 999)
                print(f"  Episodio {ep+1}:  Agente {i} a distancia {dist}")
        
        total_steps.append(steps)
    
    success_rate = success_count / 5
    avg_steps = sum(total_steps) / len(total_steps)
    
    print(f"  {phase_name}: Tasa éxito = {success_rate:.0%}, Pasos promedio = {avg_steps:.1f}")
    
    return success_rate

if __name__ == "__main__":
    print("Seleccione opción:")
    print("1. Diagnosticar modelo existente")
    print("2. Entrenar nuevo modelo enfocado")
    
    choice = input("Opción (1 o 2): ").strip()
    
    if choice == "1":
        # Diagnóstico del modelo actual
        env = analyze_agent_behavior("ppo_simple_marl")
        visualize_agent_path(env)
        
        print("\n" + "=" * 70)
        print("RECOMENDACIONES:")
        print("=" * 70)
        print("1. El agente está atascado a 1 casilla del objetivo")
        print("2. Probablemente usa demasiado WAIT (acción 0)")
        print("3. Necesita más incentivo para dar el último paso")
        print("\nSolución: Entrenar con el script de 'entrenamiento enfocado'")
        
    elif choice == "2":
        # Entrenamiento nuevo
        model, env = train_with_focused_learning()
        
        # Demostración final
        print("\n" + "=" * 70)
        print("DEMOSTRACIÓN FINAL DEL MODELO ENTRENADO")
        print("=" * 70)
        
        obs = env.reset()
        
        for step in range(30):
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            
            if step % 5 == 0:
                print(f"\nPaso {step}:")
                for i, info in enumerate(infos[:2]):
                    dist = info.get("distance_to_target", 999)
                    wp = info.get("current_waypoint", 0)
                    total_wp = info.get("total_waypoints", 1)
                    print(f"  Agente {i}: WP {wp}/{total_wp}, Dist={dist}, Recomp={rewards[i]:.2f}")
            
            if all(dones):
                print(f"\n ¡TODAS las misiones completadas en {step} pasos!")
                break
        
        print("\n ¡Modelo listo para comparar con CBS!")