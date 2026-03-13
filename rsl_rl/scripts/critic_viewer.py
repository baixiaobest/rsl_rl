#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from rsl_rl.modules.encoder_actor_critic import EncoderActorCritic
from raycaster import SceneItems, RayCaster, generate_obstacle_scan
from scipy.interpolate import RegularGridInterpolator
from height_map import SceneItems3D, HeightMapGenerator
from terrains import build_linear_stairs_scene3d, build_turning_stairs_90_scene3d, build_turning_stairs_180_scene3d

def generate_observations(x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, res=0.5, 
                         global_goal_pos=[5.0, 0.0],  # Global goal position [x, y]
                         base_lin_vel=[0.0, 0.0, 0.0], base_ang_vel=[0.0, 0.0, 0.0], 
                         joint_position=[0.0] * 12,  # Placeholder for joint positions
                         joint_velocity=[0.0] * 12,  # Placeholder for joint velocities
                         count_down=1.0,  # Placeholder for countdown
                         num_rays=15,  # Number of rays for obstacle scan
                         scene_items=None, 
                         device="cpu"):
    """Generate a grid of observations for visualization.
    
    Args:
        x_min, x_max: Range for robot x positions in the world
        y_min, y_max: Range for robot y positions in the world
        res: Resolution of the grid
        global_goal_pos: Goal position [x, y] in world frame
        base_lin_vel: Linear velocity of the base [x, y, z]
        base_ang_vel: Angular velocity of the base [x, y, z]
        scene_items: Optional SceneItems instance with obstacles
        device: Device to create tensors on
    
    Returns:
        Tuple of (observations tensor, grid info dictionary)
    """
    # Create meshgrid of robot (x, y) positions in world frame
    x = torch.arange(x_min, x_max + res, res, device=device)
    y = torch.arange(y_min, y_max + res, res, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    
    # Calculate number of grid points
    num_points = xx.numel()
    
    # Reshape meshgrid to flat arrays - these are robot positions
    robot_x = xx.reshape(-1)
    robot_y = yy.reshape(-1)
    
    # Convert global goal to local frame for each robot position
    # Local goal = global goal - robot position
    local_goal_x = torch.full_like(robot_x, global_goal_pos[0]) - robot_x
    local_goal_y = torch.full_like(robot_y, global_goal_pos[1]) - robot_y
    local_goal_z = torch.zeros_like(robot_x)  # No z component for 2D navigation
    heading_flat = torch.zeros_like(robot_x)  # No orientation for now
    
    # Create observation components
    base_lin_vel_tensor = torch.tensor(base_lin_vel, device=device).repeat(num_points, 1)  # Shape: [num_points, 3]
    base_ang_vel_tensor = torch.tensor(base_ang_vel, device=device).repeat(num_points, 1)  # Shape: [num_points, 3]
    projected_gravity = torch.tensor([0.0, 0.0, -1.0], device=device).repeat(num_points, 1)  # Shape: [num_points, 3]
    
    # Combine local goal coordinates into pose_2d_command
    pose_2d_command = torch.stack([local_goal_x, local_goal_y, local_goal_z, heading_flat], dim=1)  # Shape: [num_points, 4]
    
    # Create joint positions and velocities (12 joints with zeros)
    joint_pos = torch.tensor(joint_position, device=device).repeat(num_points, 1)  # Shape: [num_points, 12]
    joint_vel = torch.tensor(joint_velocity, device=device).repeat(num_points, 1)  # Shape: [num_points, 12]
    
    # Last actions
    actions = torch.zeros(num_points, 12, device=device)  # Shape: [num_points, 12]

    # Count down for episode length
    count_down_vec = torch.ones((num_points, 1), device=device) * count_down
    
    # Generate obstacle scans for each robot position
    if scene_items is not None:
        obstacles_scan = torch.zeros((num_points, num_rays), device=device)  # Shape: [num_points, num_rays]
        
        # Process batches to avoid excessive computation
        batch_size = 100  # Process in batches for efficiency
        for i in range(0, num_points, batch_size):
            end_idx = min(i + batch_size, num_points)
            for j in range(i, end_idx):
                robot_pos = [robot_x[j].item(), robot_y[j].item()]
                scan = generate_obstacle_scan(robot_pos, scene_items, num_rays=num_rays)
                obstacles_scan[j] = torch.tensor(scan, device=device)
    else:
        # Default: constant 10.0 distance (no obstacles)
        obstacles_scan = torch.full((num_points, num_rays), 10.0, device=device)  # Shape: [num_points, num_rays]
    
    # Concatenate all components
    observations = torch.cat([
        base_lin_vel_tensor,      # 3
        base_ang_vel_tensor,      # 3
        projected_gravity,        # 3
        pose_2d_command,          # 4
        joint_pos,                # 12
        joint_vel,                # 12
        actions,                  # 12
        count_down_vec,           # 1
        obstacles_scan,           # num_rays
    ], dim=1)
    
    print(f"Generated observation grid with shape: {observations.shape}")
    print(f"Grid dimensions: {len(x)}x{len(y)}, Total points: {num_points}")
    print(f"Global goal position: {global_goal_pos}")
    
    # Store the grid info for visualization
    grid_info = {
        'robot_positions': torch.stack([robot_x, robot_y], dim=1),
        'local_goals': torch.stack([local_goal_x, local_goal_y], dim=1),
        'grid_shape': (len(x), len(y))
    }
    
    return observations, grid_info

def generate_observations_height_map(
    x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, res=0.05,
    global_goal_pos=[5.0, 0.0],
    base_lin_vel=[0.0, 0.0, 0.0], base_ang_vel=[0.0, 0.0, 0.0],
    joint_position=[0.0] * 12, joint_velocity=[0.0] * 12,
    device="cpu",
    # Height scan parameters
    height_scan_size=21,
    height_scan_resolution=0.2,
    robot_height=0.4,
    robot_heading: float = 0.0,
    point_toward_goal=False,  # If True, heading points toward goal (overrides robot_heading)
    heading_regions=None,  # List of dicts: [{"x_min": ..., "x_max": ..., "y_min": ..., "y_max": ..., "heading": ...}, ...]
    ordering="xy",  # "xy" (x-major) or "yx" (y-major)
    scene_items_3d=None,
    hm_generator=None,
    ground_z=0.0,
):
    """Generate observations with a 21x21 height-map scan around each robot position.

    - Scan values: (robot_height - terrain_height).
    - Ordering 'xy' flattens x-major (inner loop x, outer loop y) to match GridPatternCfg semantics.
    - local_goal_z is queried from the height map at global_goal_pos (x, y).
    """
    # Grid of robot positions
    x = torch.arange(x_min, x_max + res, res, device=device)
    y = torch.arange(y_min, y_max + res, res, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    num_points = xx.numel()
    robot_x = xx.reshape(-1)
    robot_y = yy.reshape(-1)

    # Precompute height-map covering the whole region plus margin for windows
    def _build_hm(scene3d, xmin, xmax, ymin, ymax, scan_size, res_hm, ground):
        half_extent = (scan_size - 1) * 0.5 * res_hm
        top_left = (xmin - half_extent, ymax + half_extent)  # image-like convention (y decreases downward)
        size_xy = ((xmax - xmin) + 2 * half_extent, (ymax - ymin) + 2 * half_extent)
        gen = HeightMapGenerator(scene3d, ground_z=ground)
        gen.generate_map(top_left=top_left, size_xy=size_xy, resolution=res_hm)
        return gen

    if hm_generator is None:
        if scene_items_3d is None:
            scene_items_3d = SceneItems3D()  # empty scene => ground only
        hm_generator = _build_hm(scene_items_3d, x_min, x_max, y_min, y_max,
                                 height_scan_size, height_scan_resolution, ground_z)

    # Pose command and other obs components
    goal_x, goal_y = float(global_goal_pos[0]), float(global_goal_pos[1])
    local_goal_x = torch.full_like(robot_x, goal_x) - robot_x
    local_goal_y = torch.full_like(robot_y, goal_y) - robot_y

    # local_goal_z: query height at global goal position (bilinear), same for all points
    h_goal = hm_generator.query_point((goal_x, goal_y), method="bilinear")
    local_goal_z = torch.full_like(robot_x, float(h_goal), dtype=torch.float32) + robot_height

    # Compute heading: either constant or pointing toward goal
    if point_toward_goal:
        heading_flat = torch.atan2(local_goal_y, local_goal_x)
    else:
        heading_flat = torch.full_like(robot_x, robot_heading)
    
    # Apply region-specific heading overrides
    if heading_regions is not None:
        for region in heading_regions:
            x_min_r = region["x_min"]
            x_max_r = region["x_max"]
            y_min_r = region["y_min"]
            y_max_r = region["y_max"]
            heading_r = region["heading"]
            
            # Find points within this region
            mask = (robot_x >= x_min_r) & (robot_x <= x_max_r) & (robot_y >= y_min_r) & (robot_y <= y_max_r)
            heading_flat[mask] = heading_r
    
    base_lin_vel_tensor = torch.tensor(base_lin_vel, device=device).repeat(num_points, 1)
    base_ang_vel_tensor = torch.tensor(base_ang_vel, device=device).repeat(num_points, 1)
    projected_gravity = torch.tensor([0.0, 0.0, -1.0], device=device).repeat(num_points, 1)
    pose_2d_command = torch.stack([local_goal_x, local_goal_y, local_goal_z, heading_flat], dim=1)
    joint_pos = torch.tensor(joint_position, device=device).repeat(num_points, 1)
    joint_vel = torch.tensor(joint_velocity, device=device).repeat(num_points, 1)
    actions = torch.zeros(num_points, 12, device=device)
    foot_scan = torch.zeros((num_points, 4*8), device=device)  # Placeholder for foot scans

    # Simplified height scan at a robot position: snap to grid alignment, then query_grid
    def _height_scan_at(rx, ry, heading):
        width = (height_scan_size - 1) * height_scan_resolution
        height = (height_scan_size - 1) * height_scan_resolution
        rotate_offset = heading - np.pi*0.5

        # Centered top-left around robot (image-like convention)
        x0 = rx - width * 0.5
        y0 = ry + height * 0.5

        # Snap top-left to precomputed grid indices
        _, _, _, meta = hm_generator.get_full_map()
        fx0 = (x0 - meta["x_left"]) / meta["res_x"]
        fy0 = (meta["y_top"] - y0) / meta["res_y"]
        ix0 = int(round(fx0))
        iy0 = int(round(fy0))
        x0_aligned = meta["x_left"] + ix0 * meta["res_x"]
        y0_aligned = meta["y_top"] - iy0 * meta["res_y"]

        flat = hm_generator.query_grid(
            top_left=(x0_aligned, y0_aligned),
            size_xy=(width, height),
            resolution=height_scan_resolution,
            order="x-major" if ordering.lower() == "xy" else "y-major",
            return_coords=False,
            rotate_offset=rotate_offset
        ).astype(np.float32)

        point_height = hm_generator.query_point((rx, ry), method="bilinear")

        # Convert to robot-relative heights and ensure flattened 1D
        return (point_height - flat).reshape(-1)

    # Build all height scans (flattened)
    scan_len = height_scan_size * height_scan_size
    height_scans = torch.empty((num_points, scan_len), device=device, dtype=torch.float32)
    batch_size = 200
    for i in range(0, num_points, batch_size):
        end_idx = min(i + batch_size, num_points)
        for j in range(i, end_idx):
            # Use per-point heading if point_toward_goal is enabled
            heading_j = heading_flat[j].item()
            scan = _height_scan_at(robot_x[j].item(), robot_y[j].item(), heading_j)
            height_scans[j] = torch.from_numpy(scan).to(device)

    height_scans = height_scans.reshape(num_points, -1)

    # debug_scanes = _height_scan_at(0.0, 1.0, 1.5*np.pi)

    # Assemble observations
    observations = torch.cat([
        base_lin_vel_tensor,      # 3
        base_ang_vel_tensor,      # 3
        projected_gravity,        # 3
        pose_2d_command,          # 4 (z from goal height)
        joint_pos,                # 12
        joint_vel,                # 12
        actions,                  # 12
        foot_scan,                # 32 (placeholder)
        height_scans,             # 21*21 flattened
    ], dim=1)

    grid_info = {
        'robot_positions': torch.stack([robot_x, robot_y], dim=1),
        'local_goals': torch.stack([local_goal_x, local_goal_y], dim=1),
        'grid_shape': (len(x), len(y)),
    }
    return observations, grid_info

def follow_gradient_flow(values_grid, x_coords, y_coords, start_point, step_size=0.05, 
                        max_steps=1000, min_gradient_magnitude=1e-5):
    """
    Follow the gradient flow from a starting point using gradient ascent.
    
    Args:
        values_grid: 2D numpy array of value function values
        x_coords: 1D array of x coordinates
        y_coords: 1D array of y coordinates
        start_point: Starting point [x, y]
        step_size: Step size for gradient ascent
        max_steps: Maximum number of steps
        min_gradient_magnitude: Stop if gradient magnitude is below this threshold
    
    Returns:
        List of points along the flow path
    """
    # Create interpolation function for the values grid
    # RegularGridInterpolator expects (x, y) ordering for the points
    interp_func = RegularGridInterpolator((x_coords, y_coords), values_grid, 
                                         bounds_error=False, fill_value=None)
    
    # Create gradient interpolation functions
    # We'll compute the gradient of the value function with respect to x and y
    dx_grid, dy_grid = np.gradient(values_grid, x_coords, y_coords)
    dx_interp = RegularGridInterpolator((x_coords, y_coords), dx_grid, 
                                       bounds_error=False, fill_value=0.0)
    dy_interp = RegularGridInterpolator((x_coords, y_coords), dy_grid, 
                                       bounds_error=False, fill_value=0.0)
    
    # Get scalar min/max bounds for comparison
    x_min = float(x_coords.min())
    x_max = float(x_coords.max())
    y_min = float(y_coords.min())
    y_max = float(y_coords.max())
    
    # Initialize the flow path with the starting point
    path = [start_point]
    current_point = np.array(start_point)
    
    # Follow the gradient
    for step in range(max_steps):
        # Compute the gradient at the current point
        # Reshape for RegularGridInterpolator which expects 2D points
        point_for_interp = current_point.reshape(1, -1)
        
        # Get scalar gradients
        grad_x = float(dx_interp(point_for_interp)[0])
        grad_y = float(dy_interp(point_for_interp)[0])
        gradient = np.array([grad_x, grad_y])
        
        # Check if gradient magnitude is too small
        gradient_magnitude = np.linalg.norm(gradient)
        if gradient_magnitude < min_gradient_magnitude:
            break
        
        # Normalize the gradient
        if gradient_magnitude > 0:
            normalized_gradient = gradient / gradient_magnitude
        else:
            break
            
        # Update the current point
        current_point = current_point + step_size * normalized_gradient
        path.append(current_point.copy())
        
        # Check if we're out of bounds - using scalar comparisons
        if (current_point[0] < x_min or 
            current_point[0] > x_max or
            current_point[1] < y_min or 
            current_point[1] > y_max):
            break
    
    return np.array(path)

def visualize_value_heatmap(values, grid_info, global_goal_pos=None, scene_items=None, title="Value Function Heatmap", 
                           save_path=None, show_plot=True, contour_levels=20, vmin=None, vmax=None, 
                           show_vector_field=False, vector_density=20, flow_start_points=None, flow_step_size=0.05, 
                           flow_max_steps=2000):
    """Visualize the value function as a 2D heatmap over robot positions.
    
    Args:
        values: Tensor of shape [num_points] with critic values
        grid_info: Dictionary with grid information from generate_observations
        global_goal_pos: Optional global goal position to mark on the plot
        scene_items: Optional SceneItems instance with obstacles to visualize
        title: Title for the plot
        save_path: Optional path to save the figure
        show_plot: Whether to show the plot (plt.show())
        contour_levels: Number of contour levels to show
        vmin: Minimum value for color mapping (values below will be clipped)
        vmax: Maximum value for color mapping (values above will be clipped)
        show_vector_field: Whether to display a vector field showing the gradient
        vector_density: Density of the vector field (higher = fewer vectors)
        flow_start_points: List of starting points for gradient flow visualization
        flow_step_size: Step size for gradient flow
    """
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle, Circle
    import numpy as np
    
    # Extract grid information
    robot_positions = grid_info['robot_positions'].cpu().numpy()
    grid_shape = grid_info['grid_shape']
    
    # Reshape values to match the grid
    values_grid = values.cpu().reshape(grid_shape).numpy()
    
    # Create x and y coordinates for the heatmap
    x_coords = np.unique(robot_positions[:, 0])
    y_coords = np.unique(robot_positions[:, 1])
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a color normalization if vmin/vmax are specified
    norm = None
    if vmin is not None or vmax is not None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Create heatmap with value range
    mesh = ax.pcolormesh(x_coords, y_coords, values_grid.T, shading='auto', cmap='viridis', norm=norm)
    plt.colorbar(mesh, label='Value')
    
    # Add vector field showing gradient direction if requested
    if show_vector_field:
        # First transpose the values grid to match the orientation used in pcolormesh
        values_grid_for_gradient = values_grid.T
        
        # Calculate gradients on the transposed grid (to match the orientation of the heatmap)
        # np.gradient returns (gradient_y, gradient_x)
        dy_grid, dx_grid = np.gradient(values_grid_for_gradient)
        
        # Create a meshgrid for the plot - note we're NOT using indexing='ij' here
        XX, YY = np.meshgrid(x_coords, y_coords)  # default is 'xy' indexing
        
        # Subsample for clearer visualization
        x_step = max(1, len(x_coords) // vector_density)
        y_step = max(1, len(y_coords) // vector_density)
        
        # Negate gradients to point toward HIGHER values (gradient ascent)
        U = dx_grid[::y_step, ::x_step]  # Note the y_step first, then x_step to match XX/YY
        V = dy_grid[::y_step, ::x_step]
        
        # Normalize vectors for consistent length
        magnitude = np.sqrt(U**2 + V**2)
        nonzero = magnitude > 0
        U[nonzero] = U[nonzero] / magnitude[nonzero]
        V[nonzero] = V[nonzero] / magnitude[nonzero]
        
        # Plot vector field with subsampling
        ax.quiver(XX[::y_step, ::x_step], YY[::y_step, ::x_step], U, V,
                color='white', alpha=0.8, scale=25, 
                scale_units='width', pivot='mid', width=0.002,
                label='Value Gradient')
    
    # Plot gradient flow paths if start points are provided
    if flow_start_points is not None:
        for i, start_point in enumerate(flow_start_points):
            # Follow the gradient flow from this starting point
            flow_path = follow_gradient_flow(
                values_grid, x_coords, y_coords, start_point, 
                step_size=flow_step_size, max_steps=flow_max_steps
            )
            
            # Plot the flow path
            ax.plot(flow_path[:, 0], flow_path[:, 1], 'o-', 
                   color='cyan' if i % 5 != 0 else 'yellow', 
                   markersize=3, linewidth=2, alpha=0.7, zorder=5,
                   label='Gradient Flow' if i == 0 else None)
            
            # Mark the starting point
            ax.plot(start_point[0], start_point[1], 'o', 
                   color='white', markersize=6, zorder=6,
                   label='Flow Start' if i == 0 else None)
    
    # Add obstacles from scene_items if provided
    if scene_items is not None:
        # Draw boxes with red outlines, no fill
        for box in scene_items.boxes:
            x, y, width, length = box
            # Create rectangle centered at (x, y)
            rect = Rectangle((x - width/2, y - length/2), width, length, 
                           fill=False, linewidth=2.0, ec='red', 
                           zorder=10, label='Box' if 'Box' not in ax.get_legend_handles_labels()[1] else None)
            ax.add_patch(rect)
        
        # Draw circles with red outlines, no fill
        for circle in scene_items.circles:
            x, y, radius = circle
            circ = Circle((x, y), radius, fill=False, linewidth=2.0, ec='red', 
                         zorder=10, label='Circle' if 'Circle' not in ax.get_legend_handles_labels()[1] else None)
            ax.add_patch(circ)
    
    # Mark the global goal position if provided
    if global_goal_pos is not None:
        ax.plot(global_goal_pos[0], global_goal_pos[1], 'r*', markersize=15, label='Goal', zorder=15)
    
    # Add contour lines for better visualization
    if contour_levels != 0:
        contour = ax.contour(x_coords, y_coords, values_grid.T, levels=contour_levels, colors='white', alpha=0.5)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
        
        # Add labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Create legend if we have any labeled elements
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    
    # Make the plot look nice with equal aspect ratio
    ax.set_aspect('equal')
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

def visualize_value_surface(values, grid_info, global_goal_pos=None, title="Value Function 3D View", 
                           save_path=None, show_plot=True):
    """Visualize the value function as a 3D surface plot.
    
    Args:
        values: Tensor of shape [num_points] with critic values
        grid_info: Dictionary with grid information from generate_observations
        global_goal_pos: Optional global goal position to mark on the plot
        title: Title for the plot
        save_path: Optional path to save the figure
        show_plot: Whether to show the plot (plt.show())
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    
    # Extract grid information
    robot_positions = grid_info['robot_positions'].cpu().numpy()
    grid_shape = grid_info['grid_shape']
    
    # Create the figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Reshape data for 3D plot
    X = robot_positions[:, 0].reshape(grid_shape)
    Y = robot_positions[:, 1].reshape(grid_shape)
    Z = values.cpu().reshape(grid_shape)
    
    # Create the surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Value')
    
    # Mark the goal position if provided
    if global_goal_pos is not None:
        max_value = values.max().item()
        ax.scatter([global_goal_pos[0]], [global_goal_pos[1]], [max_value], 
                  color='red', s=100, marker='*', label='Goal')
        ax.legend()
    
    # Add labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Value')
    ax.set_title(title)
    
    # Save the figure if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

def main():
    # Hardcoded parameters - modify these as needed
    model_path = "logs/rsl_rl/EncoderActorCriticGO2/E2ENavigation/ObstacleScanner/model_3900.ptrom"  # Replace with your model path
    num_rays = 32
    num_critic_obs = 50 + num_rays
    num_actor_obs = 50 + num_rays
    num_actions = 12
    encoder_dims = None
    device = "cuda"  # Use "cuda" if you have a GPU
    global_goal_pos = [10, 0]
    base_lin_vel = [0.0, 0.0, 0.0]
    
    # Check if model file exists
    if not os.path.isfile(model_path):
        print(f"Error: Model file '{model_path}' not found")
        return
    
    # Load model file
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
        
    # Create model with hardcoded parameters
    model = EncoderActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        actor_hidden_dims=[128, 128, 64],
        critic_hidden_dims=[128, 128],
        # actor_hidden_dims=[128, 128, 64],
        # critic_hidden_dims=[128, 128, 64],
        # actor_hidden_dims=[128, 128, 64, 64],
        # critic_hidden_dims=[128, 128, 64, 64],
        encoder_dims=encoder_dims,
        activation="elu",
        tanh_output=True
    )
        
    # Load the state dict into the model
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Print model summary
    print("\nModel Summary:")
    print(f"Encoder: {model.encoder}")
    print(f"Critic: {model.critic}")

    # Create scene items for obstacle scanning
    scene = SceneItems()
    scene.add_circle([12.0, 2.0], 1.0)
    scene.add_circle([8.0, 5.0], 1.0)
    scene.add_box([8.0, -2.0], 2.0, 2.0)
    scene.add_box([4.0, 2.0], 0.5, 2.0)
    
    # Generate artificial observation data
    observations, grid_info = generate_observations(
        x_min=0.0,
        x_max=20.0,
        y_min=-10.0,
        y_max=10.0,
        res=0.1,
        base_lin_vel=base_lin_vel,
        global_goal_pos=global_goal_pos,  # Goal is at (10, 0) in world frame
        scene_items=scene,
        num_rays=num_rays,
        count_down=1,
        device=device
    )
    
    # Evaluate the critic
    print("\nEvaluating critic...")
    with torch.no_grad():
        values = model.evaluate(observations)
    
    # Display the results
    print("\nResults:")
    print("Observation Shape:", observations.shape)
    print("Values Shape:", values.shape)
    print("\nObservation Samples (first row):")
    print(observations[0].cpu().numpy())
    
    # Calculate statistics
    mean_value = values.mean().item()
    min_value = values.min().item()
    max_value = values.max().item()
    std_value = values.std().item()
    
    print("\nValue Statistics:")
    print(f"Mean: {mean_value:.4f}")
    print(f"Min: {min_value:.4f}")
    print(f"Max: {max_value:.4f}")
    print(f"Standard Deviation: {std_value:.4f}")

    # Define some starting points for gradient flow visualization
    flow_start_points = [
        [6.0, -5.0],
        [17.0, -5.0],
        [5.0, 5.0],
        [7.5, 1.0]
    ]

    # 2D heatmap visualization with gradient flow
    visualize_value_heatmap(
        values=values,
        grid_info=grid_info,
        global_goal_pos=global_goal_pos,
        scene_items=scene,
        title=f"Value Function with Gradient Flow",
        save_path="value_function_with_flow.png",
        show_vector_field=False,        # Show vector field
        vector_density=25,             # Control vector density
        # flow_start_points=flow_start_points,  # Add flow paths
        # flow_step_size=0.05,           # Step size for flow paths
        # flow_max_steps=10000
    )

    # 3D surface visualization
    visualize_value_surface(
        values=values,
        grid_info=grid_info,
        global_goal_pos=global_goal_pos,
        title=f"Value Function 3D View",
        save_path="value_function_3d.png"
    )

def main_height_map():
    # Configure model and scan parameters
    model_path = "logs/rsl_rl/EncoderActorCriticGO2/Stairs/CNN/model_9997_turn180.pt"  # Update as needed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_goal_pos = [2.0, -1.0]
    # global_goal_pos = [2.0, 3.5]
    base_lin_vel = [0.0, 0.0, 0.0]

    # Height-map scan settings
    height_scan_size = 21
    height_scan_resolution = 0.2
    robot_height = 0.4
    ordering = "xy"  # x-major

    # Observation dims (81 generic features + 21*21 scan)
    scan_len = height_scan_size * height_scan_size  # 441
    generic_obs = 81
    num_critic_obs = generic_obs + scan_len
    num_actor_obs = generic_obs + scan_len
    num_actions = 12

    # CNN encoder config for 21x21 scan
    e2e_cnn_config = [
        { 'type': 'reshape', 'input_size': 441, 'shape': [1, 21, 21] },
        { 'type': 'conv', 'out_channels': 8,  'kernel_size': 3, 'dilation': 1, 'stride': 1, 'padding': 1 },
        { 'type': 'pool', 'kernel_size': 2, 'stride': 2 },
        { 'type': 'conv', 'out_channels': 16, 'kernel_size': 3, 'dilation': 1, 'stride': 1, 'padding': 1 },
        { 'type': 'pool', 'kernel_size': 2, 'stride': 2 },
        { 'type': 'conv', 'out_channels': 32, 'kernel_size': 3, 'dilation': 1, 'stride': 1, 'padding': 1 },
        { 'type': 'adaptive_pool', 'output_size': (2, 2) }
    ]

    # Load model
    if not os.path.isfile(model_path):
        print(f"Error: Model file '{model_path}' not found")
        return
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model = EncoderActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        actor_hidden_dims=[128, 128, 64],
        critic_hidden_dims=[128, 128, 64],
        encoder_dims=e2e_cnn_config,
        encoder_type="cnn",
        share_encoder_with_critic=True,
        activation="elu",
        tanh_output=True,
    )
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Build 3D stairs scene (as boxes)
    # stairs_scene = build_linear_stairs_scene3d(
    #     center=(8.0, 0.0),
    #     num_steps=12,
    #     step_height=0.1,
    #     step_width=0.30,
    #     stairs_width=1.2,
    #     direction="y+",
    #     base_z=0.0,
    # )

     # Build 90-degree turning stairs scene as 3D boxes
    # stairs_scene = build_turning_stairs_90_scene3d(
    #     center=(8.0, 0.0),
    #     num_steps_run1=20,
    #     num_steps_run2=20,
    #     step_height=0.1,
    #     step_width=0.3,
    #     stairs_width=1.2,
    #     landing_length=1.2,
    #     landing_width=None,
    #     turn_right=True,
    #     base_z=0.0,
    # )

    stairs_scene = build_turning_stairs_180_scene3d(
        center=(0.0, 0.0),
        num_steps_run1=10,
        num_steps_run2=10,
        step_height=0.05,
        step_width=0.25,
        stairs_width=2.0,
        landing_length=2.0,
        landing_width=None,
        run2_on_positive_x=True,
        base_z=0.0,
    )

    # World region to evaluate
    x_min, x_max = -2.0, 4.0
    y_min, y_max = -3.0, 6.0

    # heading_regions = [
    #     {"x_min": -1.0, "x_max": 1.0, "y_min": 0.0, "y_max": 2.5, "heading": 0.5*np.pi},
    #     {"x_min": 1.0, "x_max": 3.0, "y_min": 0.0, "y_max": 2.5, "heading": -0.5*np.pi},
    # ]
    heading_regions = None

    # Generate observations using the height-map scan
    observations, grid_info = generate_observations_height_map(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        res=0.02,
        global_goal_pos=global_goal_pos,
        base_lin_vel=base_lin_vel,
        device=device,
        height_scan_size=height_scan_size,
        height_scan_resolution=height_scan_resolution,
        robot_height=robot_height,
        robot_heading=0.5*np.pi,
        point_toward_goal=False,
        heading_regions=heading_regions,
        ordering=ordering,
        scene_items_3d=stairs_scene,
        hm_generator=None,   # let the function precompute the map
        ground_z=0.0,
    )

    # Evaluate the critic
    print("\nEvaluating critic on height-map observations...")
    with torch.no_grad():
        values = model.evaluate(observations)

    # Visualizations
    visualize_value_heatmap(
        values=values,
        grid_info=grid_info,
        global_goal_pos=global_goal_pos,
        scene_items=None,
        title="Value Function (Height-Map Scan on Stairs)",
        save_path="value_function_heightmap_stairs_heatmap.png",
        show_plot=True,
        contour_levels=0,
        show_vector_field=False,
        vmin=-1.5, 
        vmax=0.8
    )

    visualize_value_surface(
        values=values,
        grid_info=grid_info,
        global_goal_pos=global_goal_pos,
        title="Value Function 3D View (Height-Map Stairs)",
        save_path="value_function_heightmap_stairs_3d.png",
        show_plot=True,
    )

if __name__ == "__main__":
    # main()
    main_height_map()  # Uncomment to run height-map version