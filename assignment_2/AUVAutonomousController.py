import rclpy
from rclpy.node import Node
import numpy as np
import math
from geometry_msgs.msg import Wrench 
from std_msgs.msg import Float64 
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry  
from rclpy.qos import qos_profile_sensor_data

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z

class PID:
    def __init__(self, kp, ki, kd, min_out, max_out):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out = min_out
        self.max_out = max_out
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, setpoint, measured, dt):
        error = setpoint - measured
        p_term = self.kp * error
        self.integral += error * dt
        self.integral = max(min(self.integral, self.max_out), self.min_out)
        i_term = self.ki * self.integral
        d_term = self.kd * (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        output = p_term + i_term + d_term
        return max(min(output, self.max_out), self.min_out)

class AUVAutonomousController(Node):
    def __init__(self):
        super().__init__('auv_autonomous_controller')

        # SEARCH PARAMETERS (Define Rectangle Here) 
        self.SEARCH_LENGTH = 10.0  # Meters (X axis)
        self.SEARCH_WIDTH  = 50.0   # Meters (Y axis)
        self.LANE_WIDTH    = 5   # Meters (Distance between search lanes)
        
        self.WAYPOINT_TOLERANCE = 0.5 # Meter (tolerans for waypoint acseptens)

        #  CONFIGURATION
        self.MAX_THRUST = 3.0 
        self.MAX_STEER_ANGLE = math.radians(15) 
        self.TARGET_ALTITUDE = 5.0
        self.MASS_LIMIT = 0.5
        self.GRAVITY_COMPENSATION = 25

        #  PID CONTROLLERS parameter
        self.pid_buoyancy = PID(kp=250.0, ki=10.0, kd=30.0, min_out=-500.0, max_out=500.0) 
        self.pid_yaw      = PID(kp=1.0, ki=0.0, kd=1.0, min_out=-self.MAX_STEER_ANGLE, max_out=self.MAX_STEER_ANGLE)
        self.pid_pitch    = PID(kp=5.0, ki=0.5, kd=1.0, min_out=-self.MAX_STEER_ANGLE, max_out=self.MAX_STEER_ANGLE)
        self.pid_mass     = PID(kp=2.0, ki=0.1, kd=0.5, min_out=-self.MASS_LIMIT, max_out=self.MASS_LIMIT)

        #  PUBLISHERS 
        self.pub_thrust      = self.create_publisher(Float64, 'cmd_thrust', 10)
        self.pub_yaw         = self.create_publisher(Float64, 'cmd_yaw', 10)
        self.pub_pitch       = self.create_publisher(Float64, 'cmd_pitch', 10)
        self.pub_moving_mass = self.create_publisher(Float64, 'cmd_moving_mass', 10)
        self.pub_ballast     = self.create_publisher(Wrench, 'cmd_ballast', 10)

        #  SUBSCRIBERS 
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.sub_imu  = self.create_subscription(Imu, '/imu', self.imu_callback, qos_profile_sensor_data)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile_sensor_data)

        #  VARIABLES 
        self.last_time = self.get_clock().now()
        self.current_pitch = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # Navigation State
        self.waypoints = self.generate_grid_waypoints()
        self.current_wp_index = 0
        self.search_finished = False
        
       


    def generate_grid_waypoints(self):
        wps = []

        
        num_lanes = int(math.ceil(self.SEARCH_WIDTH / self.LANE_WIDTH))
        
        for i in range(num_lanes):
            y_pos = i * self.LANE_WIDTH
            
            # even lane 
            if i % 2 == 0:
                wps.append((self.SEARCH_LENGTH, y_pos))
                
                if i < num_lanes - 1:
                    wps.append((self.SEARCH_LENGTH, y_pos + self.LANE_WIDTH))
            
            # odd lane 
            else:
                wps.append((0.0, y_pos))
                
                if i < num_lanes - 1:
                    wps.append((0.0, y_pos + self.LANE_WIDTH))

        # Final Waypoint: Return to Start (0,0)
        wps.append((0.0, 0.0))
        
        return wps

    def odom_callback(self, msg: Odometry):
        # Update Position
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Update Yaw
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.current_yaw = yaw

    def imu_callback(self, msg: Imu):
        q = msg.orientation
        _, pitch, _ = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.current_pitch = pitch
        
        # Moving Mass Control 
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt == 0: return 
        
        mass_pos = self.pid_mass.compute(0.0, pitch, 0.02) 
        self.pub_moving_mass.publish(Float64(data=mass_pos))

    def scan_callback(self, msg: LaserScan):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        if dt == 0: dt = 0.05

        #  BUOYANCY & MASS 
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = msg.range_max
        if len(ranges) > 0:
            altitude = np.min(ranges)
        else:
            altitude = msg.range_max
            
        pid_output = self.pid_buoyancy.compute(self.TARGET_ALTITUDE, altitude, dt)
        total_ballast = self.GRAVITY_COMPENSATION + pid_output
        
        msg_ballast = Wrench()
        msg_ballast.force.z = float(total_ballast)
        self.pub_ballast.publish(msg_ballast)
        
        # Pitch 
        cmd_pitch = self.pid_pitch.compute(0.0, self.current_pitch, dt)
        self.pub_pitch.publish(Float64(data=cmd_pitch))

        #  GRID SEARCH NAVIGATION -
        if self.search_finished:
            # STOP
            self.pub_thrust.publish(Float64(data=0.0))
            self.pub_yaw.publish(Float64(data=0.0))
            return

        # Get current target
        target_x, target_y = self.waypoints[self.current_wp_index]
        
        # Calculate Distance to Target
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        dist_error = math.sqrt(dx**2 + dy**2)
        
        # Check if reached
        if dist_error < self.WAYPOINT_TOLERANCE:
           
            self.current_wp_index += 1
            if self.current_wp_index >= len(self.waypoints):
                self.search_finished = True
                
                return
            else:
                # Update target immediately
                target_x, target_y = self.waypoints[self.current_wp_index]
                dx = target_x - self.current_x
                dy = target_y - self.current_y

        # Calculate Desired Heading (Atan2)
        desired_yaw = math.atan2(dy, dx)
        
        # Calculate Yaw Error (shortest path)
        yaw_error = desired_yaw - self.current_yaw
        while yaw_error > math.pi: yaw_error -= 2*math.pi
        while yaw_error < -math.pi: yaw_error += 2*math.pi
        
        # Compute Control
        cmd_yaw = self.pid_yaw.compute(yaw_error, 0.0, dt)

        
        # Slow down if turning hard, otherwise full speed
        turn_penalty = abs(yaw_error) * 5
        thrust_cmd = max(0.0, min(self.MAX_THRUST - turn_penalty, self.MAX_THRUST))
        
        # Publish
        self.pub_yaw.publish(Float64(data=cmd_yaw))
        self.pub_thrust.publish(Float64(data=thrust_cmd))

def main(args=None):
    rclpy.init(args=args)
    node = AUVAutonomousController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()