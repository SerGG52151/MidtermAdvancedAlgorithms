"""
Verifica que los escenarios generados sean resolubles por CBS antes de entrenar MARL.
Si no son resolubles, cambia automáticamente la semilla.
"""

import time
import json
from typing import Dict, List, Tuple, Optional
from mission_env import (
    generate_large_map, 
    generate_missions, 
    MissionMAPFEnvironment,
    Mission,
    save_scenario,
    load_scenario
)
from AStar import find_multi_goal_paths
from Grid import GridMap, Position

def scenario_to_cbs_format(grid_map: GridMap, missions: Dict[int, Mission]) -> Tuple:
    """
    Convierte un escenario de misiones al formato requerido por CBS.
    
    Returns:
        grid: List[List[int]] - 0=libre, 1=obstáculo
        starts: List[Tuple[int, int]] - posiciones iniciales
        goals_list: List[List[Tuple[int, int]]] - listas de waypoints por agente
    """
    # Convertir grid a matriz
    grid = [[0 for _ in range(grid_map.width)] for _ in range(grid_map.height)]
    for obstacle in grid_map.obstacles:
        grid[obstacle.row][obstacle.col] = 1
    
    # Convertir misiones
    starts = []
    goals_list = []
    
    for agent_id in sorted(missions.keys()):
        mission = missions[agent_id]
        starts.append((mission.start.row, mission.start.col))
        goals_list.append([(wp.row, wp.col) for wp in mission.waypoints])
    
    return grid, starts, goals_list

def validate_solution(solution, starts, goals_list):
    """Valida que una solución de CBS sea realmente válida."""
    if solution is None:
        return False, "CBS devolvió None"
    
    if not isinstance(solution, dict):
        return False, f"Solución no es dict: {type(solution)}"
    
    # Verificar que todos los agentes tengan caminos
    if len(solution) != len(starts):
        return False, f"Número de agentes incorrecto: {len(solution)} vs {len(starts)}"
    
    # Verificar cada camino
    for agent_id in range(len(starts)):
        if agent_id not in solution:
            return False, f"Agente {agent_id} no está en la solución"
        
        path = solution[agent_id]
        if not isinstance(path, list):
            return False, f"Camino del agente {agent_id} no es lista: {type(path)}"
        
        if len(path) == 0:
            return False, f"Camino del agente {agent_id} está vacío"
        
        # Verificar que empieza en la posición correcta
        if path[0] != starts[agent_id]:
            return False, f"Agente {agent_id} no empieza en {starts[agent_id]}"
        
        # Verificar que pasa por todos sus waypoints
        for waypoint in goals_list[agent_id]:
            if waypoint not in path:
                return False, f"Agente {agent_id} no pasa por waypoint {waypoint}"
    
    return True, "Solución válida"

def is_scenario_solvable(
    grid_map: GridMap,
    missions: Dict[int, Mission],
    max_cbs_time: int = 30,
    verbose: bool = True
) -> Tuple[bool, Optional[Dict], float]:
    """
    Verifica si un escenario es resolubles por CBS.
    
    Returns:
        solvable: bool - True si CBS encontró solución VÁLIDA
        solution: Dict - solución encontrada (None si no hay)
        solve_time: float - tiempo que tomó CBS
    """
    
    # Convertir a formato CBS
    grid, starts, goals_list = scenario_to_cbs_format(grid_map, missions)
    
    if verbose:
        print(f"Verificando escenario con CBS...")
        print(f"  Mapa: {grid_map.width}x{grid_map.height}")
        print(f"  Agentes: {len(missions)}")
        print(f"  Waypoints por agente: {len(goals_list[0])}")
        print(f"  Obstáculos: {len(grid_map.obstacles)}")
    
    # Ejecutar CBS con timeout
    start_time = time.time()
    
    try:
        # Usar un límite de tiempo aproximado
        solution = find_multi_goal_paths(
            starts, 
            goals_list, 
            grid, 
            max_iterations=50000,
            verbose=False  # Cambiar a True para debug
        )
        
        solve_time = time.time() - start_time
        
        # VALIDACIÓN ESTRICTA: verificar que la solución sea realmente válida
        is_valid, validation_msg = validate_solution(solution, starts, goals_list)
        
        if is_valid:
            if verbose:
                print(f"  ✓ CBS encontró solución VÁLIDA en {solve_time:.3f}s")
                # Calcular métricas
                max_len = max(len(p) for p in solution.values())
                total_cost = sum(len(p) for p in solution.values())
                print(f"  ✓ Makespan: {max_len}, Costo total: {total_cost}")
            return True, solution, solve_time
        else:
            if verbose:
                print(f"  ✗ CBS NO encontró solución válida: {validation_msg} (tiempo: {solve_time:.3f}s)")
            return False, None, solve_time
            
    except Exception as e:
        solve_time = time.time() - start_time
        if verbose:
            print(f"  ✗ Error en CBS: {e} (tiempo: {solve_time:.3f}s)")
        return False, None, solve_time

def generate_and_validate_scenario(
    width: int = 16,
    height: int = 16,
    num_agents: int = 3,
    num_waypoints: int = 4,
    obstacle_density: float = 0.25,
    min_distance: int = 8,
    base_seed: int = 42,
    max_attempts: int = 10,
    verbose: bool = True
) -> Tuple[Optional[GridMap], Optional[Dict], int, Optional[Dict]]:
    """
    Genera escenarios hasta encontrar uno que sea resolubles por CBS.
    
    Returns:
        grid_map: GridMap válido o None
        missions: Dict de misiones válidas o None  
        seed: Semilla que funcionó
        cbs_solution: Solución encontrada por CBS
    """
    
    for attempt in range(max_attempts):
        seed = base_seed + attempt
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Intento {attempt+1}/{max_attempts} - Semilla: {seed}")
            print(f"{'='*60}")
        
        try:
            # 1. Generar mapa
            grid_map = generate_large_map(
                width=width,
                height=height,
                obstacle_density=obstacle_density,
                seed=seed,
                min_free_paths=3
            )
            
            # 2. Generar misiones
            missions = generate_missions(
                grid_map,
                num_agents=num_agents,
                num_waypoints=num_waypoints,
                min_distance=min_distance,
                seed=seed * 2  # Diferente semilla para misiones
            )
            
            # 3. Verificar con CBS
            solvable, cbs_solution, solve_time = is_scenario_solvable(
                grid_map, missions, verbose=verbose
            )
            
            if solvable:
                if verbose:
                    print(f"\n ¡Escenario válido encontrado con semilla {seed}!")
                    print(f"   Tiempo CBS: {solve_time:.3f}s")
                    # Mostrar detalles de la solución
                    if cbs_solution:
                        print(f"   Detalles de solución:")
                        for agent_id, path in cbs_solution.items():
                            print(f"     Agente {agent_id}: {len(path)} pasos")
                
                # Guardar escenario para referencia
                save_scenario(f"valid_scenario_seed{seed}.json", grid_map, missions)
                
                return grid_map, missions, seed, cbs_solution
            else:
                if verbose:
                    print(f"  ✗ Escenario no válido, probando nueva semilla...")
                
        except Exception as e:
            if verbose:
                print(f"  ✗ Error generando escenario: {e}")
            continue
    
    # Si llegamos aquí, no encontramos escenario válido
    if verbose:
        print(f"\n{'!'*60}")
        print(f"ERROR: No se encontró escenario válido después de {max_attempts} intentos.")
        print(f"       Cambia los parámetros (obstacle_density, min_distance) o usa una semilla base diferente.")
        print(f"{'!'*60}")
    
    return None, None, base_seed, None

# El resto del archivo permanece igual...

def validate_and_save_scenario_for_training(
    output_file: str = "training_scenario.json",
    **kwargs
) -> bool:
    """
    Genera un escenario válido y lo guarda para entrenamiento.
    
    Returns:
        True si se encontró y guardó un escenario válido
    """
    
    grid_map, missions, seed, cbs_solution = generate_and_validate_scenario(
        **kwargs
    )
    
    if grid_map is not None and missions is not None:
        # Guardar escenario
        save_scenario(output_file, grid_map, missions)
        
        # También guardar solución CBS para referencia
        if cbs_solution is not None:
            with open(f"cbs_solution_seed{seed}.json", "w") as f:
                # Convertir a formato serializable
                sol_dict = {
                    str(aid): [(r, c) for r, c in path]
                    for aid, path in cbs_solution.items()
                }
                json.dump(sol_dict, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"ESCENARIO LISTO PARA ENTRENAMIENTO")
        print(f"{'='*60}")
        print(f"  Archivo: {output_file}")
        print(f"  Semilla: {seed}")
        print(f"  Mapa: {grid_map.width}x{grid_map.height}")
        print(f"  Agentes: {len(missions)}")
        print(f"  Waypoints por agente: {len(next(iter(missions.values())).waypoints)}")
        print(f"  Solución CBS guardada como: cbs_solution_seed{seed}.json")
        print(f"{'='*60}")
        
        return True
    else:
        return False

def load_and_validate_existing_scenario(
    scenario_file: str,
    verbose: bool = True
) -> Tuple[bool, Optional[Dict], float]:
    """
    Carga un escenario existente y verifica si sigue siendo válido.
    Útil para verificar escenarios guardados previamente.
    """
    
    try:
        grid_map, missions = load_scenario(scenario_file)
        
        if verbose:
            print(f"Cargando escenario: {scenario_file}")
        
        return is_scenario_solvable(grid_map, missions, verbose=verbose)
        
    except Exception as e:
        if verbose:
            print(f"Error cargando escenario: {e}")
        return False, None, 0.0

def analyze_scenario_difficulty(
    grid_map: GridMap,
    missions: Dict[int, Mission],
    num_samples: int = 100
) -> Dict:
    """
    Analiza la dificultad del escenario con métricas heurísticas.
    """
    
    metrics = {
        "map_size": f"{grid_map.width}x{grid_map.height}",
        "num_agents": len(missions),
        "num_waypoints": len(next(iter(missions.values())).waypoints),
        "obstacle_density": len(grid_map.obstacles) / (grid_map.width * grid_map.height),
        "free_cells": grid_map.width * grid_map.height - len(grid_map.obstacles),
        "avg_waypoint_distance": 0,
        "min_waypoint_distance": float('inf'),
        "connectivity_score": 0
    }
    
    # Calcular distancias entre waypoints
    all_positions = []
    for mission in missions.values():
        all_positions.append(mission.start)
        all_positions.extend(mission.waypoints)
    
    distances = []
    for i, pos1 in enumerate(all_positions):
        for pos2 in all_positions[i+1:]:
            dist = abs(pos1.row - pos2.row) + abs(pos1.col - pos2.col)
            distances.append(dist)
            metrics["min_waypoint_distance"] = min(metrics["min_waypoint_distance"], dist)
    
    if distances:
        metrics["avg_waypoint_distance"] = sum(distances) / len(distances)
    
    # Estimación de conectividad (BFS simple)
    from collections import deque
    
    free_cells = []
    for r in range(grid_map.height):
        for c in range(grid_map.width):
            pos = Position(r, c)
            if grid_map.is_free(pos):
                free_cells.append(pos)
    
    if free_cells:
        # Realizar BFS desde una celda aleatoria
        start = free_cells[0]
        visited = set([start])
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            for neighbor in grid_map.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        metrics["connectivity_score"] = len(visited) / len(free_cells)
    
    # Clasificar dificultad
    if metrics["obstacle_density"] < 0.2 and metrics["connectivity_score"] > 0.9:
        metrics["difficulty"] = "Fácil"
    elif metrics["obstacle_density"] < 0.3 and metrics["connectivity_score"] > 0.7:
        metrics["difficulty"] = "Medio"
    else:
        metrics["difficulty"] = "Difícil"
    
    return metrics

if __name__ == "__main__":
    # Ejemplo de uso
    
    print("=" * 70)
    print("VALIDADOR DE ESCENARIOS PARA MARL")
    print("Genera escenarios y verifica que sean resolubles por CBS")
    print("=" * 70)
    
    # Opción 1: Generar nuevo escenario válido
    success = validate_and_save_scenario_for_training(
        output_file="valid_training_scenario.json",
        width=12,  # Empezar con mapa mediano
        height=12,
        num_agents=3,
        num_waypoints=2,  # Empezar con pocos waypoints
        obstacle_density=0.2,
        min_distance=6,
        base_seed=42,
        max_attempts=5,
        verbose=True
    )
    
    if success:
        # Opción 2: Analizar dificultad del escenario generado
        from mission_env import Mission
        
        grid_map, missions = load_scenario("valid_training_scenario.json")
        
        print("\n ANÁLISIS DE DIFICULTAD:")
        metrics = analyze_scenario_difficulty(grid_map, missions)
        
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Opción 3: Verificar que el escenario guardado sigue siendo válido
        print("\n VERIFICANDO ESCENARIO GUARDADO...")
        solvable, solution, solve_time = load_and_validate_existing_scenario(
            "valid_training_scenario.json",
            verbose=True
        )
        
        if solvable:
            print(" Escenario validado correctamente")
        else:
            print(" Escenario no válido, necesita regeneración")
    else:
        print("\n No se pudo generar escenario válido.")
        print("   Intenta con:")
        print("   1. Disminuir obstacle_density (ej: 0.15)")
        print("   2. Disminuir num_waypoints (ej: 1 o 2)")
        print("   3. Disminuir min_distance (ej: 4)")
        print("   4. Cambiar base_seed (ej: 100)")