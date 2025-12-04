from __future__ import annotations

import time
import json
from typing import Dict, List, Tuple
from mission_env import load_scenario, manhattan_distance
from AStar import find_multi_goal_paths
from AStar_visualize import visualize_paths
from train_marl_missions import create_mission_scenario
from mission_wrapper import MissionMAPFVecEnv
from stable_baselines3 import PPO

def run_cbs_on_mission(scenario_file: str):
    """Ejecuta CBS en un escenario con misiones múltiples."""
    print("=" * 60)
    print("CBS - MISIONES MULTIPLES")
    print("=" * 60)
    
    # Cargar escenario
    grid_map, missions = load_scenario(scenario_file)
    
    # Convertir a formato CBS
    grid = [[0 for _ in range(grid_map.width)] for _ in range(grid_map.height)]
    for obs in grid_map.obstacles:
        grid[obs.row][obs.col] = 1
    
    starts = []
    goals_list = []
    
    for aid in sorted(missions.keys()):
        mission = missions[aid]
        starts.append((mission.start.row, mission.start.col))
        # CBS necesita todos los waypoints como goals
        goals_list.append([(wp.row, wp.col) for wp in mission.waypoints])
    
    print(f"Mapa: {grid_map.width}x{grid_map.height}")
    print(f"Agentes: {len(starts)}")
    print(f"Waypoints por agente: {len(goals_list[0])}")
    print(f"Obstáculos: {len(grid_map.obstacles)}")
    print()
    
    # Ejecutar CBS
    t0 = time.time()
    solution = find_multi_goal_paths(starts, goals_list, grid, verbose=True)
    dt = time.time() - t0
    
    if solution is None:
        print("❌ CBS no encontró solución")
        return None, dt
    
    # Calcular métricas
    max_len = max(len(p) for p in solution.values())
    total_cost = sum(len(p) for p in solution.values())
    avg_cost = total_cost / len(solution)
    
    print("\n" + "=" * 60)
    print("RESULTADOS CBS:")
    print(f"  Tiempo de búsqueda: {dt:.3f}s")
    print(f"  Makespan: {max_len} pasos")
    print(f"  Costo total: {total_cost} pasos")
    print(f"  Costo promedio: {avg_cost:.1f} pasos")
    print("=" * 60)
    
    # Visualizar
    visualize_paths(grid, starts, goals_list, solution)
    
    return solution, dt

def run_marl_on_mission(scenario_file: str, model_path: str):
    """Ejecuta modelo MARL entrenado en el mismo escenario."""
    print("\n" + "=" * 60)
    print("MARL - MISIONES MULTIPLES")
    print("=" * 60)
    
    # Cargar escenario
    grid_map, missions = load_scenario(scenario_file)
    
    # Crear entorno
    from mission_env import MissionMAPFEnvironment
    env = MissionMAPFEnvironment(grid_map, missions, max_time=200)
    
    # Cargar modelo entrenado
    model = PPO.load(model_path)
    
    # Wrapper para evaluación
    eval_env = MissionMAPFVecEnv(lambda: env, num_agents=len(missions))
    
    # Ejecutar episodio
    obs = eval_env.reset()
    total_rewards = [0.0] * len(missions)
    steps = 0
    mission_completion = {aid: False for aid in missions.keys()}
    
    t0 = time.time()
    
    while steps < 200:  # Límite de pasos
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(actions)
        
        steps += 1
        
        # Acumular recompensas
        for i in range(len(rewards)):
            total_rewards[i] += rewards[i]
        
        # Verificar completitud de misiones
        for info in infos:
            aid = info["agent_id"]
            if not mission_completion[aid] and aid in info.get("completed_agents", []):
                mission_completion[aid] = True
                print(f"  Agente {aid} completó misión en paso {steps}")
        
        if all(dones):
            break
    
    dt = time.time() - t0
    
    # Calcular métricas
    all_complete = all(mission_completion.values())
    success_rate = sum(mission_completion.values()) / len(mission_completion)
    avg_reward = sum(total_rewards) / len(total_rewards)
    
    print("\n" + "=" * 60)
    print("RESULTADOS MARL:")
    print(f"  Tiempo de ejecución: {dt:.3f}s")
    print(f"  Pasos totales: {steps}")
    print(f"  Tasa de éxito: {success_rate:.1%} ({sum(mission_completion.values())}/{len(mission_completion)})")
    print(f"  Recompensa promedio: {avg_reward:.2f}")
    print(f"  Completitud total: {'SI' if all_complete else 'NO'}")
    print("=" * 60)
    
    # Renderizar trayectorias
    env.render(
        starts={aid: missions[aid].start for aid in missions},
        goals={aid: missions[aid].waypoints[-1] for aid in missions},  # Objetivo final
        paths={aid: [env.agent_positions[aid]] for aid in missions}  # Solo posición final
    )
    
    return {
        "steps": steps,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "all_complete": all_complete,
        "time": dt
    }

def compare_cbs_vs_marl():
    """Comparación completa CBS vs MARL en el mismo escenario."""
    
    # 1. Generar/escenario compartido
    scenario_file = "comparison_scenario.json"
    
    print("Generando escenario de comparación...")
    grid_map = generate_large_map(seed=42)
    missions = generate_missions(grid_map, seed=42)
    save_scenario(scenario_file, grid_map, missions)
    
    print("\n" + "=" * 60)
    print("COMPARACIÓN CBS vs MARL")
    print("=" * 60)
    
    # 2. Ejecutar CBS
    print("\n>>> EJECUTANDO CBS <<<")
    cbs_solution, cbs_time = run_cbs_on_mission(scenario_file)
    
    # 3. Entrenar MARL rápido (para demostración)
    print("\n>>> ENTRENANDO MARL (versión rápida) <<<")
    # Nota: En producción, usarías un modelo ya entrenado
    print("(Usando modelo pre-entrenado para comparación justa)")
    
    # 4. Ejecutar MARL
    print("\n>>> EJECUTANDO MARL <<<")
    marl_results = run_marl_on_mission(scenario_file, "ppo_mission_agent_seed42")
    
    # 5. Comparación cuantitativa
    print("\n" + "=" * 60)
    print("COMPARACIÓN CUANTITATIVA")
    print("=" * 60)
    
    if cbs_solution:
        cbs_steps = max(len(p) for p in cbs_solution.values())
        print(f"\nCBS:")
        print(f"  Tiempo de planificación: {cbs_time:.3f}s")
        print(f"  Makespan: {cbs_steps} pasos")
        print(f"  Garantía: Óptimo (si encuentra solución)")
    
    print(f"\nMARL:")
    print(f"  Tiempo de ejecución: {marl_results['time']:.3f}s")
    print(f"  Pasos: {marl_results['steps']}")
    print(f"  Tasa de éxito: {marl_results['success_rate']:.1%}")
    print(f"  Completitud: {'SI' if marl_results['all_complete'] else 'NO'}")
    
    print("\n" + "=" * 60)
    print("CONCLUSIONES:")
    print("-" * 60)
    print("CBS: Mejor para planificación offline, garantía óptima")
    print("MARL: Mejor para ejecución online, adaptable a cambios")
    print("=" * 60)

if __name__ == "__main__":
    compare_cbs_vs_marl()