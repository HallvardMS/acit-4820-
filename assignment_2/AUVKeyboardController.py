import rclpy
from rclpy.node import Node
import numpy as np
import math
from geometry_msgs.msg import Wrench 
from std_msgs.msg import Float64 
from sensor_msgs.msg import LaserScan, Imu
from rclpy.qos import qos_profile_sensor_data
# the code that worked last befor the problem with the spin around the z-axis (yaw)
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

        # CONFIGURATION 
        self.MAX_THRUST = 10.0 
        self.MAX_STEER_ANGLE = math.radians(15) 
        self.TARGET_ALTITUDE = 10.0
        self.MASS_LIMIT = 0.5
        
        self.GRAVITY_COMPENSATION = 25

        # PID CONTROLLERS 
        
        #Buoyancy
        self.pid_buoyancy = PID(kp=250.0, ki=10.0, kd=30.0, min_out=-500.0, max_out=500.0) 
        
        # YAW thruster
        self.pid_yaw      = PID(kp=2.0, ki=0.0, kd=0.1, min_out=-self.MAX_STEER_ANGLE, max_out=self.MAX_STEER_ANGLE)
        
        # PITCH thruster
        self.pid_pitch    = PID(kp=5.0, ki=0.5, kd=1.0, min_out=-self.MAX_STEER_ANGLE, max_out=self.MAX_STEER_ANGLE)
        
        # MOVING MASS pitch control
        self.pid_mass     = PID(kp=2.0, ki=0.1, kd=0.5, min_out=-self.MASS_LIMIT, max_out=self.MASS_LIMIT)

        # PUBLISHERS 
        self.pub_thrust      = self.create_publisher(Float64, 'cmd_thrust', 10)
        self.pub_yaw         = self.create_publisher(Float64, 'cmd_yaw', 10)
        self.pub_pitch       = self.create_publisher(Float64, 'cmd_pitch', 10)
        self.pub_moving_mass = self.create_publisher(Float64, 'cmd_moving_mass', 10)
        
        self.pub_ballast     = self.create_publisher(Wrench, 'cmd_ballast', 10)
        

        #  SUBSCRIBERS 
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.sub_imu  = self.create_subscription(Imu,       '/imu' , self.imu_callback, qos_profile_sensor_data)

        self.last_time = self.get_clock().now()
        self.current_pitch = 0.0

    def imu_callback(self, msg: Imu):
        q = msg.orientation
        _, pitch, _ = euler_from_quaternion(q.x, q.y, q.z, q.w)

        self.current_pitch = pitch

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9

        if dt == 0: return
        self.last_time = current_time
        mass_pos = self.pid_mass.compute(0.0, pitch, dt)
        self.pub_moving_mass.publish(Float64(data=mass_pos))

    def scan_callback(self, msg: LaserScan):
        dt = 0.05 
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = msg.range_max
        ranges[np.isnan(ranges)] = msg.range_max

        # ALTITUDE LOGIC 
        if len(ranges) > 0:
            altitude = np.min(ranges)
        else:
            altitude = msg.range_max
        
        # Calculate Buoyancy
        pid_output = self.pid_buoyancy.compute(self.TARGET_ALTITUDE, altitude, dt)
        total_ballast = self.GRAVITY_COMPENSATION + pid_output
        
        # Publish
        msg_ballast = Wrench()
        msg_ballast.force.z = float(total_ballast)
        self.pub_ballast.publish(msg_ballast)

        # STEERING LOGIC 
        mid_start = int(len(ranges)*0.4)
        mid_end   = int(len(ranges)*0.6)
        horizontal_slice = ranges[mid_start:mid_end]

        if len(horizontal_slice) > 0:
            max_idx = np.argmax(horizontal_slice)
            center_idx = len(horizontal_slice) // 2
            error_idx = max_idx - center_idx
            yaw_error = (error_idx / len(horizontal_slice)) * 2.0
            
            cmd_yaw = self.pid_yaw.compute(yaw_error, 0.0, dt)
            cmd_pitch = self.pid_pitch.compute(0.0, self.current_pitch, dt)
        else:
            cmd_yaw = 0.0
            cmd_pitch = 0.0
            yaw_error = 0.0

        self.pub_yaw.publish(Float64(data=cmd_yaw))
        self.pub_pitch.publish(Float64(data=cmd_pitch))

        # THRUST LOGIC
        turn_penalty = abs(yaw_error) * 5.0
        thrust = max(0.0, min(self.MAX_THRUST - turn_penalty, self.MAX_THRUST))
        self.pub_thrust.publish(Float64(data=thrust))

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