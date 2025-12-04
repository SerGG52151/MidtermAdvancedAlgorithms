"""
Verifica que CBS pueda resolver el MISMO problema que MARL (visitas secuenciales).
"""

import time
from validate_scenarios import scenario_to_cbs_format, validate_solution
from AStar import find_multi_goal_paths

def test_simple_mission():
    """Crea un problema simple y verifica que CBS pueda resolverlo."""
    
    # Crear un grid simple 5x5 sin obstáculos
    grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    
    # 2 agentes con visitas secuenciales (como lo hará MARL)
    starts = [(0, 0), (4, 4)]  # Agente 0 arriba-izquierda, Agente 1 abajo-derecha
    goals_list = [
        [(2, 2), (4, 0)],  # Agente 0: va al centro, luego abajo-izquierda
        [(2, 2), (0, 4)]   # Agente 1: va al centro, luego arriba-derecha
    ]
    
    print("=" * 60)
    print("TEST SIMPLE: CBS para visitas secuenciales")
    print("=" * 60)
    print(f"Grid: 5x5 sin obstáculos")
    print(f"Agente 0: {starts[0]} → {goals_list[0]}")
    print(f"Agente 1: {starts[1]} → {goals_list[1]}")
    print()
    
    # Ejecutar CBS
    start_time = time.time()
    solution = find_multi_goal_paths(
        starts, goals_list, grid, 
        max_iterations=10000, verbose=True
    )
    solve_time = time.time() - start_time
    
    print(f"\nTiempo CBS: {solve_time:.3f}s")
    
    if solution is None:
        print(" CBS NO encontró solución")
        return False
    
    # Validar la solución
    is_valid, msg = validate_solution(solution, starts, goals_list)
    
    if is_valid:
        print(" CBS encontró solución VÁLIDA")
        print(f"Mensaje: {msg}")
        
        # Mostrar caminos
        for agent_id, path in solution.items():
            print(f"\nAgente {agent_id} ({len(path)} pasos):")
            print(f"  Empieza: {path[0]}")
            print(f"  Termina: {path[-1]}")
            
            # Verificar waypoints
            for waypoint in goals_list[agent_id]:
                if waypoint in path:
                    print(f"  ✓ Visita: {waypoint}")
                else:
                    print(f"  ✗ NO visita: {waypoint}")
        
        return True
    else:
        print(f" Solución no válida: {msg}")
        return False

def test_mission_comparison():
    """Verifica que CBS y MARL tendrán el MISMO problema a resolver."""
    
    print("\n" + "=" * 60)
    print("COMPARACIÓN CBS vs MARL - MISMO PROBLEMA")
    print("=" * 60)
    
    # Definir el MISMO problema para ambos
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],  # Algunos obstáculos
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    
    starts = [(0, 0), (4, 4)]
    goals_list = [
        [(4, 0)],  # Agente 0: solo 1 waypoint
        [(0, 4)]   # Agente 1: solo 1 waypoint
    ]
    
    print("\nProblema común para CBS y MARL:")
    print("-" * 40)
    print(f"Mapa: 5x5 con algunos obstáculos")
    print(f"Agente 0: {starts[0]} → {goals_list[0]}")
    print(f"Agente 1: {starts[1]} → {goals_list[1]}")
    print(f"Tipo: Visitas secuenciales (NO round trips)")
    print()
    
    # CBS lo resuelve
    print("CBS resolviendo...")
    cbs_solution = find_multi_goal_paths(
        starts, goals_list, grid, verbose=False
    )
    
    if cbs_solution:
        print(" CBS puede resolver este problema")
        
        # Mostrar cómo MARL verá esto
        print("\nMARL verá esto como:")
        print("-" * 40)
        for agent_id in range(len(starts)):
            print(f"Agente {agent_id}:")
            print(f"  Start: {starts[agent_id]}")
            print(f"  Waypoints: {goals_list[agent_id]}")
            print(f"  Comportamiento esperado:")
            print(f"    1. Ir de {starts[agent_id]} a {goals_list[agent_id][0]}")
            print(f"    2. Quedarse en {goals_list[agent_id][0]} (misión completa)")
        
        return True
    else:
        print(" CBS NO puede resolver este problema")
        print("   Necesitamos hacerlo más simple...")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("VERIFICANDO COMPATIBILIDAD CBS-MARL")
    print("=" * 60)
    
    # Test 1: Problema simple
    test1 = test_simple_mission()
    
    # Test 2: Comparación
    test2 = test_mission_comparison()
    
    if test1 and test2:
        print("\n" + "=" * 60)
        print(" LISTO PARA ENTRENAR MARL")
        print("=" * 60)
        print("CBS y MARL resolverán el MISMO tipo de problema:")
        print("- Visitas secuenciales (no round trips)")
        print("- Cada agente va de start a waypoint1 a waypoint2...")
        print("- Termina en el último waypoint")
    else:
        print("\n" + "=" * 60)
        print(" AJUSTES NECESARIOS")
        print("=" * 60)