import numpy as np

class KinematicToGridWrapper:
    def __init__(self):
        # Define grid parameters
        self.x_range = (-30, 90)  # meters relative to ego
        self.y_range = (-10, 10)   # meters relative to ego

        # Calculate grid size based on 1m per cell
        self.grid_size = (
            self.x_range[1] - self.x_range[0],  # 120 cells for x
            self.y_range[1] - self.y_range[0]   # 20 cells for y
        )

        # Car dimensions
        self.car_length = 5  # meters
        self.car_width = 2   # meters

    def get_car_footprint(self, x, y, heading):
        """Calculate which cells a car occupies given its center and heading"""
        occupied_cells = []

        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        corners_car = [
            (-self.car_length/2, -self.car_width/2),
            (self.car_length/2, -self.car_width/2),
            (self.car_length/2, self.car_width/2),
            (-self.car_length/2, self.car_width/2)
        ]

        corners_world = [
            (x + dx*cos_h - dy*sin_h, y + dx*sin_h + dy*cos_h)
            for dx, dy in corners_car
        ]

        min_x = min(x[0] for x in corners_world)
        max_x = max(x[0] for x in corners_world)
        min_y = min(x[1] for x in corners_world)
        max_y = max(x[1] for x in corners_world)

        for cell_x in range(int(min_x), int(max_x) + 1):
            for cell_y in range(int(min_y), int(max_y) + 1):
                if self.point_in_rotated_rect(
                    cell_x + 0.5, cell_y + 0.5,
                    x, y, heading,
                    self.car_length, self.car_width
                ):
                    occupied_cells.append((cell_x, cell_y))

        return occupied_cells

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int(x - self.x_range[0])
        # Flip y-axis to maintain correct orientation
        grid_y = int(self.grid_size[1] - (y - self.y_range[0]) - 1)
        return grid_x, grid_y

    def point_in_rotated_rect(self, px, py, rect_x, rect_y, rect_angle, length, width):
        dx = px - rect_x
        dy = py - rect_y

        cos_h = np.cos(-rect_angle)
        sin_h = np.sin(-rect_angle)

        rotated_x = dx * cos_h - dy * sin_h
        rotated_y = dx * sin_h + dy * cos_h

        return (abs(rotated_x) <= length/2) and (abs(rotated_y) <= width/2)

    def process_observation(self, obs, left_bound, right_bound):
        """
        Process vehicle observations and return separate ego info and occupancy grid
        obs: list of [x, y, vx, vy, heading] for each vehicle (ego first)
        """
        # Extract ego vehicle state
        ego_x, ego_y, ego_vx, ego_vy, ego_heading = obs[0]

        # Initialize grid
        grid = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.float32)

        to_left =  ego_y - left_bound
        to_right = right_bound - ego_y

        left = int(self.grid_size[1]/2 -1 - to_left)
        right = int(self.grid_size[1]/2 +1 + to_right)

        if left >= 0:
            grid[:, :left + 1, 0] = 2
            grid[:, :left + 1, 2] = ego_vy
        if right < self.grid_size[1]:
            grid[:, right:, 0] = 2
            grid[:, right:, 2] = ego_vy

        # Place ego vehicle
        ego_cells = self.get_car_footprint(0, 0, ego_heading)
        for cell_x, cell_y in ego_cells:
            grid_x, grid_y = self.world_to_grid(cell_x, cell_y)

            if (0 <= grid_x < self.grid_size[0] and
                0 <= grid_y < self.grid_size[1]):
                grid[grid_x, grid_y, 0] = 1
                grid[grid_x, grid_y, 1] = 0
                grid[grid_x, grid_y, 2] = 0

        # Process other vehicles
        for vehicle in obs[1:]:
            x, y, vx, vy, heading = vehicle

            # Get relative position
            rel_x = x - ego_x
            rel_y = y - ego_y

            # Get relative velocities
            rel_vx = vx - ego_vx
            rel_vy = vy - ego_vy

            # Get relative heading

            # Skip if vehicle center is out of range
            if (rel_x < self.x_range[0] or rel_x > self.x_range[1] or
                rel_y < self.y_range[0] or rel_y > self.y_range[1]):
                continue

            # Get all cells occupied by this vehicle
            occupied_cells = self.get_car_footprint(rel_x, rel_y, heading)

            # Convert to grid coordinates and update grid
            for cell_x, cell_y in occupied_cells:
                grid_x, grid_y = self.world_to_grid(cell_x, cell_y)

                if (0 <= grid_x < self.grid_size[0] and
                    0 <= grid_y < self.grid_size[1]):
                    grid[grid_x, grid_y, 0] = 2
                    grid[grid_x, grid_y, 1] = rel_vx
                    grid[grid_x, grid_y, 2] = rel_vy

        return np.array(grid).transpose(2,0,1)
converter = KinematicToGridWrapper()