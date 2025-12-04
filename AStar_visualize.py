import matplotlib.pyplot as plt
import numpy as np

def visualize_paths(grid, starts, goals_list, paths):
    '''
    Visualizes the grid and the paths taken by each agent on a toroidal grid.
    Each round trip to a service is drawn as a separate path segment.
    Handles wrapping by drawing segments that cross boundaries properly.
    '''
    grid = np.array(grid)
    rows, cols = grid.shape

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_facecolor('#f7f7f7')
    obstacle_mask = (grid == 1)
    if obstacle_mask.any():
        ax.imshow(
            obstacle_mask, cmap='Greys', origin='upper',
            alpha=0.35, extent=[-0.5, cols-0.5, rows-0.5, -0.5]
        )

    agent_ids_sorted = sorted(paths.keys())
    cmap = plt.cm.get_cmap('tab10', len(agent_ids_sorted))
    colors = {aid: cmap(i) for i, aid in enumerate(agent_ids_sorted)}

    legend_labels: set[str] = set()

    for agent_id, full_path in paths.items():
        color = colors[agent_id]
        start_pos = starts[agent_id]
        agent_goals = goals_list[agent_id]
        
        path_segments = []
        current_idx = 0
        
        for goal in agent_goals:
            segment_start = current_idx
            found_goal = False
            
            for i in range(current_idx, len(full_path)):
                if full_path[i] == goal:
                    found_goal = True
                    for j in range(i + 1, len(full_path)):
                        if full_path[j] == start_pos:
                            path_segments.append(full_path[segment_start:j+1])
                            current_idx = j
                            break
                    break
            
            if not found_goal:
                break
        
        for seg_idx, segment in enumerate(path_segments):
            alpha = max(0.4, 0.9 - (seg_idx * 0.15))
            linewidth = max(1.4, 2.8 - (seg_idx * 0.3))
            
            for i in range(len(segment) - 1):
                row1, col1 = segment[i]
                row2, col2 = segment[i + 1]
                
                x1, y1 = col1, row1
                x2, y2 = col2, row2
                
                dx = x2 - x1
                dy = y2 - y1
                
                wrap_x = abs(dx) > cols / 2
                wrap_y = abs(dy) > rows / 2
                
                if wrap_x or wrap_y:
                    if wrap_x and not wrap_y:
                        if dx > 0:
                            # Wrapping from right to left (e.g. 0 -> 7)
                            ax.plot(
                                [x1, -0.5], [y1, y1],
                                color=color, linewidth=linewidth,
                                linestyle='--', alpha=alpha
                            )
                            ax.plot(
                                [cols - 0.5, x2], [y2, y2],
                                color=color, linewidth=linewidth,
                                linestyle='--', alpha=alpha
                            )
                        else:
                            # Wrapping from left to right (e.g. 7 -> 0)
                            ax.plot(
                                [x1, cols - 0.5], [y1, y1],
                                color=color, linewidth=linewidth,
                                linestyle='--', alpha=alpha
                            )
                            ax.plot(
                                [-0.5, x2], [y2, y2],
                                color=color, linewidth=linewidth,
                                linestyle='--', alpha=alpha
                            )
                    
                    elif wrap_y and not wrap_x:
                        if dy > 0:
                            # Wrapping from bottom to top (e.g. 0 -> 7)
                            ax.plot(
                                [x1, x1], [y1, -0.5],
                                color=color, linewidth=linewidth,
                                linestyle='--', alpha=alpha
                            )
                            ax.plot(
                                [x2, x2], [rows - 0.5, y2],
                                color=color, linewidth=linewidth,
                                linestyle='--', alpha=alpha
                            )
                        else:
                            # Wrapping from top to bottom (e.g. 7 -> 0)
                            ax.plot(
                                [x1, x1], [y1, rows - 0.5],
                                color=color, linewidth=linewidth,
                                linestyle='--', alpha=alpha
                            )
                            ax.plot(
                                [x2, x2], [-0.5, y2],
                                color=color, linewidth=linewidth,
                                linestyle='--', alpha=alpha
                            )
                    
                    else:
                        ax.plot(
                            [x1, x2], [y1, y2],
                            color=color, linewidth=linewidth,
                            linestyle=':', alpha=alpha * 0.7
                        )
                else:
                    ax.plot(
                        [x1, x2], [y1, y2],
                        color=color, linewidth=linewidth, alpha=alpha
                    )
        
        start_row, start_col = start_pos
        home_label = f'Agent {agent_id} home'
        if home_label not in legend_labels:
            ax.scatter(
                [start_col], [start_row],
                color=color, marker='o', s=220,
                edgecolors='black', linewidths=2.5, zorder=10,
                label=home_label
            )
            legend_labels.add(home_label)
        else:
            ax.scatter(
                [start_col], [start_row],
                color=color, marker='o', s=220,
                edgecolors='black', linewidths=2.5, zorder=10,
            )
        
        service_label = f'Agent {agent_id} services'
        for goal_idx, goal in enumerate(agent_goals):
            goal_row, goal_col = goal
            if service_label not in legend_labels:
                ax.scatter(
                    [goal_col], [goal_row],
                    color=color, marker='*', s=180,
                    edgecolors='black', linewidths=1.8,
                    zorder=9, alpha=0.9,
                    label=service_label
                )
                legend_labels.add(service_label)
            else:
                ax.scatter(
                    [goal_col], [goal_row],
                    color=color, marker='*', s=180,
                    edgecolors='black', linewidths=1.8,
                    zorder=9, alpha=0.9,
                )
            ax.text(
                goal_col + 0.12, goal_row - 0.18,
                str(goal_idx + 1),
                fontsize=9, color='black', weight='bold',
                ha='left', va='center', zorder=11,
            )

    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.grid(which='both', color='lightgray', linestyle='-', linewidth=0.6)

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))

    ax.set_xlim(-0.5, cols-0.5)
    ax.set_ylim(rows-0.5, -0.5)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('Column (wraps around at edges)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Row (wraps around at edges)', fontsize=11, fontweight='bold')
    ax.set_title(
        'Multi-Agent Paths on Toroidal Grid',
        fontsize=14, fontweight='bold', pad=14
    )

    ax.legend(
        bbox_to_anchor=(1.02, 0.5), loc='center left',
        fontsize=9, frameon=False, title='Legend'
    )

    plt.tight_layout()
    plt.show()
