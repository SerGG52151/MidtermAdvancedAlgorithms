import gymnasium as gym
from stable_baselines3 import PPO
from Grid import GridMap, MAPFEnvironment, Position
from gym_wrapper import SingleAgentWrapper
import os

def main():
    # Configurar el escenario (Un mapa simple para empezar)
    # "." = libre, "#" = obstáculo
    ascii_map = [
        ".......",
        ".###...",
        ".......",
        "...###.",
        "......."
    ]
    grid = GridMap.from_ascii(ascii_map)
    
    # Inicio (0,0) -> Meta (0, 6)
    starts = {0: Position(0, 0)}
    goals = {0: Position(4, 6)} # Esquina opuesta

    # 2. Inicializar entorno y conectar el Wrapper
    env_raw = MAPFEnvironment(grid, starts, goals, max_time=100)
    env = SingleAgentWrapper(env_raw)

    # 3. Crear el Modelo de IA
    # "MlpPolicy" es una red neuronal estándar (Multi Layer Perceptron)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)

    print("--- Iniciando Entrenamiento ---")
    
    # 4. Entrenar por X pasos de tiempo
    # 20,000 pasos es poco, pero suficiente para ver si funciona el código
    model.learn(total_timesteps=20000)
    
    print("--- Entrenamiento Finalizado ---")

    # 5. Guardar el modelo
    model.save("ppo_mapf_agent")
    print("Modelo guardado como 'ppo_mapf_agent.zip'")

    # 6. (Opcional) Probar el agente entrenado
    obs, _ = env.reset()
    print("\nProbando solución:")
    for _ in range(20):
        action, _states = model.predict(obs)
        action = int(action)
        obs, reward, done, truncated, info = env.step(action)
        
        # Renderizado simple en texto
        print(f"Acción: {action}, Recompensa: {reward}")
        
        if done:
            print("¡Meta alcanzada!")
            break

if __name__ == "__main__":
    main()