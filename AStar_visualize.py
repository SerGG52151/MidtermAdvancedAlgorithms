import matplotlib.pyplot as plt
import numpy as np

def visualize_paths(grid, starts, goals_list, paths):
    '''
    Visualizes the grid and the paths taken by each agent on a toroidal grid.
    Each round trip to a service is drawn as a separate path segment.
    Handles wrapping by drawing segments that cross boundaries properly.
    
    Args:
        grid: 2D grid array
        starts: List of starting positions for each agent
        goals_list: List of goal lists for each agent (e.g., [[(7,1), (7,3), (7,5)], ...])
        paths: Dict mapping agent_id to complete path
    '''
    grid = np.array(grid)
    rows, cols = grid.shape
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(grid, cmap='Greys', origin='upper', alpha=0.2, extent=[-0.5, cols-0.5, rows-0.5, -0.5])

    colors = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow']
    
    for agent_id, full_path in paths.items():
        color = colors[agent_id % len(colors)]
        start_pos = starts[agent_id]
        agent_goals = goals_list[agent_id]  # Get this agent's goals
        
        # Split the path into segments (one for each round trip to a service)
        # Each segment goes: start -> service -> start
        path_segments = []
        current_idx = 0
        
        for goal in agent_goals:
            # Find where this goal appears in the path
            segment_start = current_idx
            found_goal = False
            
            for i in range(current_idx, len(full_path)):
                if full_path[i] == goal:
                    found_goal = True
                    # Continue until we return to start
                    for j in range(i + 1, len(full_path)):
                        if full_path[j] == start_pos:
                            # Found complete round trip
                            path_segments.append(full_path[segment_start:j+1])
                            current_idx = j
                            break
                    break
            
            if not found_goal:
                break
        
        # Draw each segment separately
        for seg_idx, segment in enumerate(path_segments):
            alpha = 0.9 - (seg_idx * 0.15)  # Slightly fade later segments
            linewidth = 3.0 - (seg_idx * 0.3)
            
            # Draw the segment with proper toroidal wrapping
            for i in range(len(segment) - 1):
                row1, col1 = segment[i]
                row2, col2 = segment[i + 1]
                
                # For plotting: x = col, y = row
                x1, y1 = col1, row1
                x2, y2 = col2, row2
                
                # Calculate differences
                dx = x2 - x1  # column difference
                dy = y2 - y1  # row difference
                
                # Check if wrapping occurs
                wrap_x = abs(dx) > cols / 2
                wrap_y = abs(dy) > rows / 2
                
                if wrap_x or wrap_y:
                    # This is a wrapped move - draw dashed segments
                    
                    if wrap_x and not wrap_y:
                        # Wrapping in x direction (horizontal/column wrapping)
                        if dx > 0:  # wrapping from right to left
                            ax.plot([x1, cols - 0.5], [y1, y1], color=color, linewidth=linewidth, 
                                   linestyle='--', alpha=alpha)
                            ax.plot([-0.5, x2], [y2, y2], color=color, linewidth=linewidth, 
                                   linestyle='--', alpha=alpha)
                        else:  # wrapping from left to right
                            ax.plot([x1, -0.5], [y1, y1], color=color, linewidth=linewidth, 
                                   linestyle='--', alpha=alpha)
                            ax.plot([cols - 0.5, x2], [y2, y2], color=color, linewidth=linewidth, 
                                   linestyle='--', alpha=alpha)
                    
                    elif wrap_y and not wrap_x:
                        # Wrapping in y direction (vertical/row wrapping)
                        if dy > 0:  # wrapping from bottom to top
                            ax.plot([x1, x1], [y1, rows - 0.5], color=color, linewidth=linewidth, 
                                   linestyle='--', alpha=alpha)
                            ax.plot([x2, x2], [-0.5, y2], color=color, linewidth=linewidth, 
                                   linestyle='--', alpha=alpha)
                        else:  # wrapping from top to bottom
                            ax.plot([x1, x1], [y1, -0.5], color=color, linewidth=linewidth, 
                                   linestyle='--', alpha=alpha)
                            ax.plot([x2, x2], [rows - 0.5, y2], color=color, linewidth=linewidth, 
                                   linestyle='--', alpha=alpha)
                    
                    else:
                        # Wrapping in both directions (diagonal wrap)
                        ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, 
                               linestyle=':', alpha=alpha * 0.7)
                else:
                    # Normal (non-wrapped) segment
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=alpha)
        
        # Mark agent start position with large circle
        start_row, start_col = start_pos
        ax.scatter([start_col], [start_row], color=color, marker='o', s=250, 
                  edgecolors='black', linewidths=3, zorder=10, 
                  label=f'Agent {agent_id} home')
        
        # Mark each service location with a star
        for goal_idx, goal in enumerate(agent_goals):
            goal_row, goal_col = goal
            ax.scatter([goal_col], [goal_row], color=color, marker='*', s=200,
                      edgecolors='black', linewidths=2, zorder=9, alpha=0.8,
                      label=f'Agent {agent_id} service {goal_idx + 1}')

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xlim(-0.5, cols-0.5)
    ax.set_ylim(rows-0.5, -0.5)
    ax.set_xlabel('Column (wraps around at edges)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Row (wraps around at edges)', fontsize=13, fontweight='bold')
    ax.set_title('Multi-Agent Paths on Toroidal Grid\n' +
                'Each colored line shows one round trip: home → service → home\n' +
                '(Dashed lines show edge wrapping)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
    plt.tight_layout()
    plt.show()
