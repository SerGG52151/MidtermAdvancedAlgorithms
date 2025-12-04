from __future__ import annotations

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from Grid import GridMap, Position, MAPFEnvironment

@dataclass
class Mission:
    """Define una misión con múltiples waypoints (visitas secuenciales, NO round trips)."""
    start: Position
    waypoints: List[Position]  # Waypoints a visitar en secuencia
    current_waypoint_idx: int = 0
    
    @property
    def current_target(self) -> Position:
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        # Si ya visitó todos, el target es el último waypoint (queda ahí)
        return self.waypoints[-1] if self.waypoints else self.start
    
    @property
    def is_complete(self) -> bool:
        return self.current_waypoint_idx >= len(self.waypoints)
    
    def advance(self) -> bool:
        """Avanza al siguiente waypoint, retorna True si la misión está completa."""
        self.current_waypoint_idx += 1
        return self.is_complete

    def reset(self) -> None:
        """Resetea la misión al estado inicial (volver al primer waypoint)."""
        self.current_waypoint_idx = 0


class MissionMAPFEnvironment(MAPFEnvironment):
    """
    Extensión de MAPFEnvironment que maneja misiones con múltiples waypoints.
    Cada agente tiene una misión con varios puntos a visitar en secuencia.
    """
    
    def __init__(
        self,
        grid_map: GridMap,
        missions: Dict[int, Mission],
        max_time: Optional[int] = None,
    ):
        # Extraer starts (primer waypoint) y goals inicial (siguiente waypoint)
        starts = {aid: mission.start for aid, mission in missions.items()}
        goals = {aid: mission.current_target for aid, mission in missions.items()}
        
        super().__init__(grid_map, starts, goals, max_time)
        self.missions = missions
        self.completed_agents: Set[int] = set()
    
    def reset(self) -> Dict[int, Position]:
        self.time = 0
        self.completed_agents.clear()
        
        # Resetear todas las misiones
        for mission in self.missions.values():
            mission.reset()
        
        # Posiciones iniciales
        self.agent_positions = {aid: mission.start for aid, mission in self.missions.items()}
        
        # Actualizar objetivos iniciales
        self.goals = {aid: mission.current_target for aid, mission in self.missions.items()}
        
        return dict(self.agent_positions)
    
    def step(
        self,
        joint_action: Dict[int, int],
    ) -> Tuple[Dict[int, Position], bool, Dict]:
        # Paso normal del entorno base
        positions, done, info = super().step(joint_action)
        
        # Verificar si agentes alcanzaron sus waypoints actuales
        for agent_id in self.agent_positions:
            if agent_id in self.completed_agents:
                continue
                
            mission = self.missions[agent_id]
            
            # Si el agente alcanzó su waypoint actual
            if self.agent_positions[agent_id] == mission.current_target:
                # Avanzar al siguiente waypoint (visita secuencial, NO regresa)
                mission_complete = mission.advance()
                
                if mission_complete:
                    self.completed_agents.add(agent_id)
                    # Agente se queda en el último waypoint
                else:
                    # Actualizar objetivo al siguiente waypoint
                    self.goals[agent_id] = mission.current_target
        
        # Episodio termina cuando todos completan sus misiones o timeout
        all_complete = len(self.completed_agents) == len(self.missions)
        done = done or all_complete
        
        info["missions_complete"] = all_complete
        info["completed_agents"] = list(self.completed_agents)
        
        return positions, done, info
    
    def get_mission_status(self, agent_id: int) -> Tuple[int, int]:
        """Retorna (waypoint_actual, total_waypoints)."""
        mission = self.missions[agent_id]
        return mission.current_waypoint_idx, len(mission.waypoints)

def generate_large_map(
    width: int = 16,
    height: int = 16,
    obstacle_density: float = 0.25,
    seed: int = 42,
    min_free_paths: int = 3
) -> GridMap:
    """
    Genera un mapa procedural con garantía de conectividad.
    
    Args:
        width, height: Dimensiones del mapa
        obstacle_density: Probabilidad de que una celda sea obstáculo (0-1)
        seed: Semilla para reproducibilidad
        min_free_paths: Mínimo de caminos libres verticales/horizontales
    """
    random.seed(seed)
    
    # Inicializar mapa vacío
    grid = [[0 for _ in range(width)] for _ in range(height)]
    
    # 1. Generar obstáculos aleatorios
    for r in range(height):
        for c in range(width):
            if random.random() < obstacle_density:
                grid[r][c] = 1
    
    # 2. Garantizar caminos libres mínimos
    # Caminos verticales libres
    for _ in range(min_free_paths):
        col = random.randint(0, width-1)
        for r in range(height):
            grid[r][col] = 0
    
    # Caminos horizontales libres
    for _ in range(min_free_paths):
        row = random.randint(0, height-1)
        for c in range(width):
            grid[row][c] = 0
    
    # 3. Convertir a formato GridMap
    ascii_map = []
    for row in grid:
        line = ''.join('#' if cell == 1 else '.' for cell in row)
        ascii_map.append(line)
    
    return GridMap.from_ascii(ascii_map, allow_diagonal=False, allow_wait=True)

def generate_missions(
    grid_map: GridMap,
    num_agents: int = 3,
    num_waypoints: int = 3,  # Incluyendo objetivo final
    min_distance: int = 8,
    max_distance: Optional[int] = None, # <--- NUEVO PARÁMETRO
    seed: int = 42
) -> Dict[int, Mission]:
    """
    Genera misiones aleatorias pero válidas.
    Soporta Curriculum Learning mediante max_distance.
    
    Args:
        grid_map: Mapa generado
        num_agents: Número de agentes
        num_waypoints: Puntos por misión
        min_distance: Distancia mínima entre puntos
        max_distance: Distancia MÁXIMA (para niveles fáciles)
        seed: Semilla
    """
    if seed is not None:
        random.seed(seed)
    
    # Obtener todas las celdas libres
    free_cells = []
    for r in range(grid_map.height):
        for c in range(grid_map.width):
            pos = Position(r, c)
            if grid_map.is_free(pos):
                free_cells.append(pos)
    
    missions = {}
    
    for agent_id in range(num_agents):
        # Seleccionar posiciones únicas para esta misión
        selected_positions = []
        
        # 1. Elegir Inicio
        start_pos = random.choice(free_cells)
        selected_positions.append(start_pos)
        
        # 2. Elegir Waypoints secuencialmente
        for i in range(num_waypoints):
            prev_pos = selected_positions[-1]
            attempts = 0
            
            # Buscamos un candidato válido
            candidates = []
            
            # Optimización: En lugar de probar al azar, filtramos primero si es posible
            # (Esto es lento en mapas gigantes, pero rápido en 16x16)
            if max_distance is not None:
                # Solo considerar celdas en el rango [min, max]
                possible_targets = [
                    cell for cell in free_cells 
                    if min_distance <= manhattan_distance(prev_pos, cell) <= max_distance
                ]
            else:
                # Solo considerar celdas >= min_distance
                possible_targets = [
                    cell for cell in free_cells 
                    if manhattan_distance(prev_pos, cell) >= min_distance
                ]

            if possible_targets:
                next_pos = random.choice(possible_targets)
                selected_positions.append(next_pos)
            else:
                # Fallback si no hay candidatos válidos (ej. mapa muy denso)
                # Elegimos cualquiera que no sea el mismo punto
                fallback = random.choice(free_cells)
                selected_positions.append(fallback)
        
        # Crear misión
        start = selected_positions[0]
        waypoints = selected_positions[1:]
        
        missions[agent_id] = Mission(start=start, waypoints=waypoints)
    
    return missions

def manhattan_distance(pos1: Position, pos2: Position) -> int:
    return abs(pos1.row - pos2.row) + abs(pos1.col - pos2.col)

def save_scenario(filename: str, grid_map: GridMap, missions: Dict[int, Mission]):
    """Guarda un escenario para reproducibilidad."""
    import json
    
    scenario = {
        "width": grid_map.width,
        "height": grid_map.height,
        "obstacles": [(p.row, p.col) for p in grid_map.obstacles],
        "missions": {}
    }
    
    for aid, mission in missions.items():
        scenario["missions"][aid] = {
            "start": (mission.start.row, mission.start.col),
            "waypoints": [(p.row, p.col) for p in mission.waypoints]
        }
    
    with open(filename, 'w') as f:
        json.dump(scenario, f, indent=2)
    
    print(f"Escenario guardado en {filename}")

def load_scenario(filename: str) -> Tuple[GridMap, Dict[int, Mission]]:
    """Carga un escenario guardado."""
    import json
    
    with open(filename, 'r') as f:
        scenario = json.load(f)
    
    # Reconstruir mapa
    obstacles = [Position(r, c) for r, c in scenario["obstacles"]]
    grid_map = GridMap(
        width=scenario["width"],
        height=scenario["height"],
        obstacles=obstacles
    )
    
    # Reconstruir misiones
    missions = {}
    for aid_str, mission_data in scenario["missions"].items():
        aid = int(aid_str)
        start = Position(*mission_data["start"])
        waypoints = [Position(*wp) for wp in mission_data["waypoints"]]
        missions[aid] = Mission(start=start, waypoints=waypoints)
    
    return grid_map, missions