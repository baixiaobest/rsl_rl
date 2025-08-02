import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

class SceneItems:
    """Store and manage 2D geometries (boxes and circles) for obstacle simulation."""
    
    def __init__(self):
        """Initialize empty lists for boxes and circles."""
        self.boxes = []  # Each box is [pos_x, pos_y, width, length]
        self.circles = []  # Each circle is [pos_x, pos_y, radius]
    
    def add_box(self, position, width, length):
        """
        Add a box to the scene.
        
        Args:
            position: [x, y] center position of the box
            width: Size along x-axis
            length: Size along y-axis
        """
        self.boxes.append([position[0], position[1], width, length])
        return self
    
    def add_circle(self, position, radius):
        """
        Add a circle to the scene.
        
        Args:
            position: [x, y] center position of the circle
            radius: Radius of the circle
        """
        self.circles.append([position[0], position[1], radius])
        return self
    
    def clear(self):
        """Clear all items from the scene."""
        self.boxes = []
        self.circles = []
    
    def visualize(self, ax=None, robot_pos=None):
        """
        Visualize the scene items.
        
        Args:
            ax: Matplotlib axis for plotting
            robot_pos: Optional robot position to mark
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot boxes
        for box in self.boxes:
            x, y, width, length = box
            # Create rectangle centered at (x, y)
            rect = Rectangle((x - width/2, y - length/2), width, length, 
                           fill=True, alpha=0.5, color='blue', ec='black')
            ax.add_patch(rect)
        
        # Plot circles
        for circle in self.circles:
            x, y, radius = circle
            circ = Circle((x, y), radius, fill=True, alpha=0.5, color='green', ec='black')
            ax.add_patch(circ)
        
        # Mark robot position if provided
        if robot_pos is not None:
            ax.plot(robot_pos[0], robot_pos[1], 'ro', markersize=8, label='Robot')
        
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        
        return ax

class RayCaster:
    """Cast rays against scene items and compute distances."""
    
    def __init__(self, min_angle=-np.pi, max_angle=np.pi, num_rays=16, max_distance=10.0):
        """
        Initialize the ray caster.
        
        Args:
            min_angle: Minimum angle for ray casting (radians)
            max_angle: Maximum angle for ray casting (radians)
            num_rays: Number of rays to cast
            max_distance: Maximum distance for ray detection
        """
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.num_rays = num_rays
        self.max_distance = max_distance
        
        # Pre-compute ray angles
        self.angles = np.linspace(min_angle, max_angle, num_rays, endpoint=False)
    
    def cast_rays(self, robot_position, scene_items):
        """
        Cast rays from robot position against scene items.
        
        Args:
            robot_position: [x, y] position of the robot
            scene_items: SceneItems instance containing obstacles
            
        Returns:
            Array of distances for each ray
        """
        distances = np.full(self.num_rays, self.max_distance)
        
        # Check if robot is inside any obstacle
        if self._is_inside_obstacle(robot_position, scene_items):
            return np.zeros(self.num_rays)
        
        # Cast rays
        for i, angle in enumerate(self.angles):
            # Compute ray direction
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            
            # Check intersections with boxes
            for box in scene_items.boxes:
                dist = self._ray_box_intersection(robot_position, ray_dir, box)
                if dist is not None and dist < distances[i]:
                    distances[i] = dist
            
            # Check intersections with circles
            for circle in scene_items.circles:
                dist = self._ray_circle_intersection(robot_position, ray_dir, circle)
                if dist is not None and dist < distances[i]:
                    distances[i] = dist
        
        return distances
    
    def _is_inside_obstacle(self, position, scene_items):
        """Check if position is inside any obstacle."""
        # Check boxes
        for box in scene_items.boxes:
            x, y, width, length = box
            if (abs(position[0] - x) <= width/2 and
                abs(position[1] - y) <= length/2):
                return True
        
        # Check circles
        for circle in scene_items.circles:
            x, y, radius = circle
            dx = position[0] - x
            dy = position[1] - y
            if dx*dx + dy*dy <= radius*radius:
                return True
        
        return False
    
    def _ray_box_intersection(self, origin, direction, box):
        """
        Compute intersection of ray with box.
        
        Args:
            origin: Ray origin [x, y]
            direction: Ray direction [dx, dy]
            box: [x, y, width, length]
            
        Returns:
            Distance to intersection or None if no intersection
        """
        x, y, width, length = box
        
        # Box corners
        x_min, y_min = x - width/2, y - length/2
        x_max, y_max = x + width/2, y + length/2
        
        # Ray-slab intersection method
        t_near = -np.inf
        t_far = np.inf
        
        # Check x slab
        if abs(direction[0]) < 1e-6:  # Ray parallel to x axis
            if origin[0] < x_min or origin[0] > x_max:
                return None
        else:
            t1 = (x_min - origin[0]) / direction[0]
            t2 = (x_max - origin[0]) / direction[0]
            
            if t1 > t2:
                t1, t2 = t2, t1
            
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)
            
            if t_near > t_far or t_far < 0:
                return None
        
        # Check y slab
        if abs(direction[1]) < 1e-6:  # Ray parallel to y axis
            if origin[1] < y_min or origin[1] > y_max:
                return None
        else:
            t1 = (y_min - origin[1]) / direction[1]
            t2 = (y_max - origin[1]) / direction[1]
            
            if t1 > t2:
                t1, t2 = t2, t1
            
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)
            
            if t_near > t_far or t_far < 0:
                return None
        
        # Return nearest positive intersection
        return t_near if t_near > 0 else t_far
    
    def _ray_circle_intersection(self, origin, direction, circle):
        """
        Compute intersection of ray with circle.
        
        Args:
            origin: Ray origin [x, y]
            direction: Ray direction [dx, dy]
            circle: [x, y, radius]
            
        Returns:
            Distance to intersection or None if no intersection
        """
        x, y, radius = circle
        
        # Vector from ray origin to circle center
        oc = np.array([origin[0] - x, origin[1] - y])
        
        # Quadratic equation coefficients
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - radius*radius
        
        # Discriminant
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return None
        
        # Compute solutions
        t1 = (-b - np.sqrt(discriminant)) / (2*a)
        t2 = (-b + np.sqrt(discriminant)) / (2*a)
        
        # Return nearest positive intersection
        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return None
    
    def visualize(self, robot_position, scene_items, distances=None, ax=None):
        """
        Visualize ray casts against scene items.
        
        Args:
            robot_position: [x, y] position of the robot
            scene_items: SceneItems instance
            distances: Optional pre-computed distances
            ax: Matplotlib axis for plotting
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Visualize scene items
        scene_items.visualize(ax, robot_position)
        
        # Compute distances if not provided
        if distances is None:
            distances = self.cast_rays(robot_position, scene_items)
        
        # Plot rays
        for i, angle in enumerate(self.angles):
            # Ray direction
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # Ray endpoint
            end_x = robot_position[0] + distances[i] * dx
            end_y = robot_position[1] + distances[i] * dy
            
            # Plot ray
            ax.plot([robot_position[0], end_x], [robot_position[1], end_y], 'r-', alpha=0.5)
        
        return ax

def generate_obstacle_scan(robot_position, scene_items, num_rays=15, max_distance=10.0):
    """
    Generate obstacle scan for a given robot position.
    
    Args:
        robot_position: [x, y] position of the robot
        scene_items: SceneItems instance with obstacles
        num_rays: Number of rays to cast
        max_distance: Maximum distance for ray detection
        
    Returns:
        Array of distances for each ray
    """
    ray_caster = RayCaster(min_angle=-np.pi, max_angle=np.pi, num_rays=num_rays, max_distance=max_distance)
    distances = ray_caster.cast_rays(robot_position, scene_items)
    return distances

def main():
    """Test and visualize the ray casting functionality."""
    import matplotlib.pyplot as plt
    
    # Create a test scene with various obstacles
    scene = SceneItems()
    
    # Add some boxes
    scene.add_box([5.0, 0.0], 2.0, 3.0)
    scene.add_box([0.0, 4.0], 1.5, 1.5)
    scene.add_box([-3.0, -2.0], 2.0, 4.0)
    
    # Add some circles
    scene.add_circle([3.0, 3.0], 1.0)
    scene.add_circle([-4.0, 2.0], 1.2)
    scene.add_circle([4.0, -3.0], 1.5)
    
    # Test multiple robot positions
    test_positions = [
        [0.0, 0.0],  # Center
        [2.0, 2.0],  # Near a circle
        [-2.0, -3.0],  # Near a box
        [6.0, 0.5],  # Inside a box (should show all zeros)
    ]
    
    # Create a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    axs = axs.flatten()
    
    # Test each position
    for i, pos in enumerate(test_positions):
        ray_caster = RayCaster(num_rays=36, max_distance=10.0)  # More rays for better visualization
        distances = ray_caster.cast_rays(pos, scene)
        
        # Visualize
        ax = axs[i]
        ray_caster.visualize(pos, scene, distances, ax)
        inside = ray_caster._is_inside_obstacle(pos, scene)
        ax.set_title(f"Robot at {pos} {'(inside obstacle)' if inside else ''}")
        
        # Set consistent view limits
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
    
    plt.tight_layout()
    plt.savefig("raycaster_visualization.png", dpi=150)
    plt.show()
    
    # Test a single position with detailed analysis
    print("\nDetailed ray analysis for robot at (2, 2):")
    pos = [2.0, 2.0]
    ray_caster = RayCaster(num_rays=15, max_distance=10.0)
    distances = ray_caster.cast_rays(pos, scene)
    
    # Print ray distances
    for i, (angle, dist) in enumerate(zip(ray_caster.angles, distances)):
        angle_deg = angle * 180 / np.pi
        print(f"Ray {i}: angle={angle_deg:.1f}°, distance={dist:.2f}")
    
    # Visualize with a single larger plot
    plt.figure(figsize=(10, 10))
    ray_caster.visualize(pos, scene, distances)
    plt.title(f"Detailed Ray Analysis - Robot at {pos}")
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.savefig("raycaster_detailed.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()