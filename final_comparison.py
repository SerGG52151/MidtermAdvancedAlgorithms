# final_comparison.py
"""
Comparación final entre CBS y MARL usando escenarios validados.
"""

import time
import json
from validate_scenarios import generate_and_validate_scenario, load_scenario
from benchmark_missions import run_cbs_on_mission, run_marl_on_mission

def run_fair_comparison(
    num_scenarios: int = 5,
    difficulty: str = "medium",  # "easy", "medium", "hard"
    marl_model_path: str = "ppo_final_validated"
):
    """
    Ejecuta comparación justa entre CBS y MARL en múltiples escenarios.
    """
    
    print("=" * 70)
    print("COMPARACIÓN JUSTA CBS vs MARL")
    print(f"Escenarios: {num_scenarios}, Dificultad: {difficulty}")
    print("=" * 70)
    
    # Parámetros según dificultad
    if difficulty == "easy":
        params = {"width": 8, "height": 8, "num_waypoints": 1, "obstacle_density": 0.15}
    elif difficulty == "medium":
        params = {"width": 12, "height": 12, "num_waypoints": 2, "obstacle_density": 0.2}
    else:  # hard
        params = {"width": 16, "height": 16, "num_waypoints": 4, "obstacle_density": 0.25}
    
    params.update({
        "num_agents": 3,
        "verbose": False
    })
    
    results = {
        "cbs": {"successes": 0, "total_time": 0, "total_steps": 0, "scenarios": []},
        "marl": {"successes": 0, "total_time": 0, "total_steps": 0, "scenarios": []}
    }
    
    for scenario_num in range(num_scenarios):
        print(f"\n{'='*70}")
        print(f"ESCENARIO {scenario_num + 1}/{num_scenarios}")
        print(f"{'='*70}")
        
        # Generar escenario válido
        seed = 1000 + scenario_num * 100
        grid_map, missions, used_seed, _ = generate_and_validate_scenario(
            base_seed=seed,
            **params
        )
        
        if grid_map is None:
            print(f" No se pudo generar escenario válido para semilla {seed}")
            continue
        
        # Guardar escenario temporal
        scenario_file = f"comparison_scenario_{scenario_num}.json"
        from mission_env import save_scenario
        save_scenario(scenario_file, grid_map, missions)
        
        print(f"Semilla: {used_seed}, Mapa: {grid_map.width}x{grid_map.height}")
        
        # 1. Ejecutar CBS
        print(f"\n>>> CBS <<<")
        cbs_start = time.time()
        cbs_solution, cbs_time = run_cbs_on_mission(scenario_file)
        cbs_total_time = time.time() - cbs_start
        
        if cbs_solution:
            results["cbs"]["successes"] += 1
            results["cbs"]["total_time"] += cbs_total_time
            
            # Calcular makespan de CBS
            cbs_makespan = max(len(p) for p in cbs_solution.values())
            results["cbs"]["total_steps"] += cbs_makespan
            
            print(f"   CBS: {cbs_total_time:.2f}s, {cbs_makespan} pasos")
        else:
            print(f"   CBS: Falló")
        
        # 2. Ejecutar MARL
        print(f"\n>>> MARL <<<")
        marl_results = run_marl_on_mission(scenario_file, marl_model_path)
        
        if marl_results["all_complete"]:
            results["marl"]["successes"] += 1
            results["marl"]["total_time"] += marl_results["time"]
            results["marl"]["total_steps"] += marl_results["steps"]
            
            print(f"   MARL: {marl_results['time']:.2f}s, {marl_results['steps']} pasos")
        else:
            print(f"   MARL: Falló (completitud: {marl_results['success_rate']:.1%})")
        
        # Almacenar resultados del escenario
        scenario_result = {
            "seed": used_seed,
            "cbs_success": cbs_solution is not None,
            "cbs_time": cbs_total_time if cbs_solution else None,
            "cbs_steps": cbs_makespan if cbs_solution else None,
            "marl_success": marl_results["all_complete"],
            "marl_time": marl_results["time"],
            "marl_steps": marl_results["steps"],
            "marl_success_rate": marl_results["success_rate"]
        }
        
        results["cbs"]["scenarios"].append(scenario_result)
        results["marl"]["scenarios"].append(scenario_result)
    
    # Mostrar resultados agregados
    print(f"\n{'='*70}")
    print("RESULTADOS AGREGADOS")
    print(f"{'='*70}")
    
    for method in ["cbs", "marl"]:
        method_data = results[method]
        num_scenarios_tested = len(method_data["scenarios"])
        
        if num_scenarios_tested > 0:
            success_rate = method_data["successes"] / num_scenarios_tested
            
            avg_time = (method_data["total_time"] / method_data["successes"] 
                       if method_data["successes"] > 0 else 0)
            
            avg_steps = (method_data["total_steps"] / method_data["successes"] 
                        if method_data["successes"] > 0 else 0)
            
            print(f"\n{method.upper()}:")
            print(f"  Tasa de éxito: {success_rate:.1%} ({method_data['successes']}/{num_scenarios_tested})")
            print(f"  Tiempo promedio: {avg_time:.2f}s")
            print(f"  Pasos promedio: {avg_steps:.1f}")
    
    # Guardar resultados detallados
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Resultados detallados guardados en 'comparison_results.json'")
    
    # Análisis comparativo
    print(f"\n{'='*70}")
    print("ANÁLISIS COMPARATIVO")
    print(f"{'='*70}")
    
    cbs_success = results["cbs"]["successes"]
    marl_success = results["marl"]["successes"]
    
    if cbs_success > marl_success:
        print(f"CBS fue más confiable ({cbs_success} vs {marl_success} éxitos)")
    elif marl_success > cbs_success:
        print(f"MARL fue más confiable ({marl_success} vs {cbs_success} éxitos)")
    else:
        print(f"Ambos métodos igualmente confiables ({cbs_success} éxitos cada uno)")
    
    if results["cbs"]["successes"] > 0 and results["marl"]["successes"] > 0:
        cbs_avg_time = results["cbs"]["total_time"] / results["cbs"]["successes"]
        marl_avg_time = results["marl"]["total_time"] / results["marl"]["successes"]
        
        if marl_avg_time < cbs_avg_time:
            print(f"MARL fue más rápido ({marl_avg_time:.2f}s vs {cbs_avg_time:.2f}s)")
        else:
            print(f"CBS fue más rápido ({cbs_avg_time:.2f}s vs {marl_avg_time:.2f}s)")

if __name__ == "__main__":
    # Ejecutar comparación en dificultad media
    run_fair_comparison(
        num_scenarios=3,  # Empieza con pocos escenarios para prueba
        difficulty="medium",
        marl_model_path="ppo_final_validated"
    )