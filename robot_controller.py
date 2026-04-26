from controller import Robot, Motor, DistanceSensor, GPS
import math
import heapq
from collections import deque

class Node:
    def __init__(self, x, y, g=0, h=0, parent=None):
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

class AStarPlanner:
    def __init__(self, grid_size=20, cell_size=0.5):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.obstacle_map = [[False] * grid_size for _ in range(grid_size)]
        self.setup_static_obstacles()
    
    def setup_static_obstacles(self):
        center = self.grid_size // 2
        # Mark static obstacles on grid
        obstacles_world = [(-1.5, -1.5), (1.5, -1.5), (0, 1.5)]
        for ox, oy in obstacles_world:
            grid_x = int((ox + center * self.cell_size) / self.cell_size)
            grid_y = int((oy + center * self.cell_size) / self.cell_size)
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.obstacle_map[grid_y][grid_x] = True
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = grid_x + dx, grid_y + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            self.obstacle_map[ny][nx] = True
    
    def world_to_grid(self, x, y):
        center = self.grid_size // 2
        grid_x = int((x + center * self.cell_size) / self.cell_size)
        grid_y = int((y + center * self.cell_size) / self.cell_size)
        return max(0, min(grid_x, self.grid_size-1)), max(0, min(grid_y, self.grid_size-1))
    
    def grid_to_world(self, gx, gy):
        center = self.grid_size // 2
        x = (gx - center) * self.cell_size
        y = (gy - center) * self.cell_size
        return x, y
    
    def heuristic(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)
    
    def is_valid(self, x, y):
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size and not self.obstacle_map[y][x]
    
    def plan(self, start_x, start_y, goal_x, goal_y):
        start_gx, start_gy = self.world_to_grid(start_x, start_y)
        goal_gx, goal_gy = self.world_to_grid(goal_x, goal_y)
        
        open_list = []
        closed_set = set()
        start_node = Node(start_gx, start_gy, 0, self.heuristic(start_gx, start_gy, goal_gx, goal_gy))
        heapq.heappush(open_list, start_node)
        
        while open_list:
            current = heapq.heappop(open_list)
            
            if (current.x, current.y) in closed_set:
                continue
            
            closed_set.add((current.x, current.y))
            
            if current.x == goal_gx and current.y == goal_gy:
                path = []
                node = current
                while node is not None:
                    world_x, world_y = self.grid_to_world(node.x, node.y)
                    path.append((world_x, world_y))
                    node = node.parent
                return path[::-1]
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                nx, ny = current.x + dx, current.y + dy
                
                if not self.is_valid(nx, ny) or (nx, ny) in closed_set:
                    continue
                
                g = current.g + (1.414 if abs(dx) + abs(dy) == 2 else 1)
                h = self.heuristic(nx, ny, goal_gx, goal_gy)
                
                neighbor = Node(nx, ny, g, h, current)
                heapq.heappush(open_list, neighbor)
        
        return []

class RobotNavigator:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        self.gps = self.robot.getGPS("gps")
        self.gps.enable(self.timestep)
        
        self.sensors = {}
        sensor_names = ["ds_front", "ds_front_left", "ds_front_right", "ds_left", "ds_right"]
        for name in sensor_names:
            sensor = self.robot.getDistanceSensor(name)
            sensor.enable(self.timestep)
            self.sensors[name] = sensor
        
        self.left_motor = self.robot.getMotor("left wheel motor")
        self.right_motor = self.robot.getMotor("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        self.planner = AStarPlanner()
        self.path = []
        self.current_waypoint_idx = 0
        self.max_speed = 6.0
        self.obstacle_detected = False
        self.step_count = 0
        self.last_replan_step = 0
    
    def get_position(self):
        gps_data = self.gps.getValues()
        return gps_data[0], gps_data[2]
    
    def check_obstacles(self):
        threshold = 0.5
        for name, sensor in self.sensors.items():
            if sensor.getValue() < threshold:
                return True
        return False
    
    def calculate_angle_to_target(self, current_x, current_y, target_x, target_y):
        dx = target_x - current_x
        dy = target_y - current_y
        angle = math.atan2(dy, dx)
        return angle
    
    def move_towards_waypoint(self, target_x, target_y):
        current_x, current_y = self.get_position()
        distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        
        if distance < 0.15:
            return True
        
        target_angle = self.calculate_angle_to_target(current_x, current_y, target_x, target_y)
        
        left_speed = self.max_speed
        right_speed = self.max_speed
        
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        
        return False
    
    def plan_path(self):
        current_x, current_y = self.get_position()
        self.path = self.planner.plan(current_x, current_y, 3, 3)
        self.current_waypoint_idx = 0
        print(f"[PLAN] New path planned with {len(self.path)} waypoints from ({current_x:.2f}, {current_y:.2f}) to (3.0, 3.0)")
    
    def stop_motors(self):
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
    
    def run(self):
        print("[INIT] Robot Controller Started")
        print("[INIT] Planning initial path...")
        self.plan_path()
        
        while self.robot.step(self.timestep) != -1:
            self.step_count += 1
            current_x, current_y = self.get_position()
            
            obstacle_detected = self.check_obstacles()
            
            if self.step_count - self.last_replan_step > 10:
                if obstacle_detected or (self.path and current_x < -4 or current_x > 4 or current_y < -4 or current_y > 4):
                    print(f"[OBSTACLE] Detected at step {self.step_count}, replanning...")
                    self.plan_path()
                    self.last_replan_step = self.step_count
            
            if not self.path:
                print("[ERROR] No path available!")
                self.stop_motors()
                break
            
            if self.current_waypoint_idx >= len(self.path):
                print(f"[SUCCESS] Goal reached at ({current_x:.2f}, {current_y:.2f})")
                self.stop_motors()
                break
            
            target_x, target_y = self.path[self.current_waypoint_idx]
            
            waypoint_reached = self.move_towards_waypoint(target_x, target_y)
            
            if waypoint_reached:
                self.current_waypoint_idx += 1
                print(f"[WAYPOINT] Reached waypoint {self.current_waypoint_idx}/{len(self.path)} at ({current_x:.2f}, {current_y:.2f})")
            
            if self.step_count % 100 == 0:
                print(f"[STATUS] Step {self.step_count}, Position: ({current_x:.2f}, {current_y:.2f}), Waypoint: {self.current_waypoint_idx}/{len(self.path)}")

if __name__ == "__main__":
    navigator = RobotNavigator()
    navigator.run()