import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import sys

import numpy as np

from std_msgs.msg import Int64, Float32, Float32MultiArray
from geometry_msgs.msg import Twist

### https://github.com/IvayloAsenov/Wall-Follower-Robot-ROS/blob/master/wall_follower.py

class PID_angle:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.curr_error = 0
        self.prev_error = 0
        self.sum_error = 0
        self.prev_error_deriv = 0
        self.curr_error_deriv = 0
        self.control = 0
        self.dt = dt
        
    def update_control(self, current_error, reset_prev=False):
        
        self.prev_error = self.curr_error
        self.curr_error = current_error
        
        #Calculating the integral error
        self.sum_error = self.sum_error + self.curr_error*self.dt

        #Calculating the derivative error
        self.curr_error_deriv = (self.curr_error - self.prev_error) / self.dt

        #Calculating the PID Control
        self.control = self.Kp * self.curr_error + self.Ki * self.sum_error + self.Kd * self.curr_error_deriv


    def get_control(self):
        return self.control
    
class PID_dist:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.curr_error = 0
        self.prev_error = 0
        self.sum_error = 0
        self.prev_error_deriv = 0
        self.curr_error_deriv = 0
        self.control = 0
        self.dt = dt
        
    def update_control(self, current_error, reset_prev=False):
        
        self.prev_error = self.curr_error
        self.curr_error = current_error
        
        #Calculating the integral error
        self.sum_error = self.sum_error + self.curr_error*self.dt

        #Calculating the derivative error
        self.curr_error_deriv = (self.curr_error - self.prev_error) / self.dt

        #Calculating the PID Control
        self.control = self.Kp * self.curr_error + self.Ki * self.sum_error + self.Kd * self.curr_error_deriv


    def get_control(self):
        return self.control

class VelocityGenerator(Node):

    def __init__(self):		
		# Creates the node.
        super().__init__('velocity_generator')

        self.angle_controller = PID_angle(1, 1, 1, 1)
        self.dist_controller = PID_dist(1, 1, 1, 1)

        #Set up QoS Profiles for passing numbers over WiFi
        num_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

        #Set up QoS Profiles for passing numbers over WiFi
        vel_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.RELIABLE,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

        self.dist_and_angle_subscriber = self.create_subscription(
				Float32MultiArray,
				'/object_detect/dist_and_angle',
                self.dist_and_angle_callback,
				num_qos_profile)
        self.dist_and_angle_subscriber # Prevents unused variable warning.

        # create velocity publisher
        self.velocity_publisher = self.create_publisher(
				Twist, 
				'/cmd_vel',
				vel_qos_profile)
        self.velocity_publisher

        self.target_distance = 0.4 # meters

        self.distance = None
        self.angle = None

        self.linear_x_vel = None
        self.angular_z_vel = None

    def dist_and_angle_callback(self, msg):
        self.distance, self.angle = msg.data
        self.get_logger().info(f'received distance: {self.distance}, received angle: {self.angle}')

    def get_turn_direction(self):

        noise = 10

        if noise < self.angle < 180:
            self.direction = 1 # left
        
        elif noise < self.angle < (360 - noise):
            self.direction = -1 # right

        else:
            self.direction = 0

    # def get_angle_error(self):

    #     noise = 2 # deg

    #     if (self.angle < noise) or (self.angle > (360-noise)):
    #         angle_error = 0

    #     elif self.angle < 180:
    #         angle_error = self.angle - noise

    #     else:
    #         angle_error  = self.angle - 360 + noise

    #     return angle_error

    def get_angular_velocity(self):

        ### should call PID here?
        
        self.get_turn_direction()
        speed = 1
        self.angular_z_vel = float(self.direction * speed)

    def get_linear_velocity(self):

        ### should call PID here?

        noise = 0.01

        diff = self.target_distance - self.distance

        if diff > noise:
            self.linear_x_vel = -0.1

        elif diff < -noise:
            self.linear_x_vel = 0.1

        else:
            self.linear_x_vel = 0.0

        self.linear_x_vel = float(self.linear_x_vel)
        self.get_logger().info(f'linear velocity: {self.linear_x_vel}')	

    def publish_spin_velocity(self):
        if self.distance is not None:
            self.get_angular_velocity()
            self.get_linear_velocity()
            vel = Twist()
            vel.linear.x = self.linear_x_vel
            vel.angular.z = self.angular_z_vel
            self.velocity_publisher.publish(vel)
            self.get_logger().info(f'velocity: {vel}')	

    def get_user_input(self):
        return self._user_input

def main():
    rclpy.init() #init routine needed for ROS2.
    velocity_generator = VelocityGenerator() #Create class object to be used.
    
    while rclpy.ok():

        try:
            rclpy.spin_once(velocity_generator) # Trigger callback processing.
            velocity_generator.publish_spin_velocity()

        except KeyboardInterrupt:
            break

	#Clean up and shutdown.

    velocity_generator.destroy_node()  
    rclpy.shutdown()


if __name__ == '__main__':
	main()