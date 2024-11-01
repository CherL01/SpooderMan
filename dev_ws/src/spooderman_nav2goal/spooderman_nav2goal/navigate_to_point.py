import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import sys

import numpy as np

from std_msgs.msg import Int64, Float32, Float32MultiArray
from geometry_msgs.msg import PointStamped

### https://github.com/IvayloAsenov/Wall-Follower-Robot-ROS/blob/master/wall_follower.py

class PointPublisher(Node):

    def __init__(self):		
		# Creates the node.
        super().__init__('point_publisher')

        #Set up QoS Profiles for passing numbers over WiFi
        num_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

        self.clicked_point_subscriber = self.create_subscription(
				PointStamped,
				'/clicked_point',
                self.clicked_point_callback,
				num_qos_profile)
        self.clicked_point_subscriber # Prevents unused variable warning.

        self.velocity_publisher = self.create_publisher(
				Twist, 
				'/cmd_vel',
				vel_qos_profile)
        self.velocity_publisher

        self.target_distance = 0.35 # meters
        self.target_angle = 0 # deg

        self.distance = None
        self.angle = None

        self.linear_x_vel = None
        self.angular_z_vel = None

        self.pos_top_speed = 0.21 # m/s
        self.neg_top_speed = -0.21
        self.Kp_dist = 3
        self.Kp_angle = 0.08

        self.pos_top_angular_speed = 2.8 # rad/s
        self.neg_top_angular_speed = -2.8

    def dist_and_angle_callback(self, msg):
        self.distance, self.angle = msg.data
        self.get_logger().info(f'received distance: {self.distance}, received angle: {self.angle}')

    def get_turn_direction_and_angle_diff(self):

        noise = 5

        if noise < self.angle < 180:
            self.direction = 1 # left
            angle_diff = self.angle - noise
        
        elif noise < self.angle < (360 - noise):
            self.direction = -1 # right
            angle_diff = (360 - noise) - self.angle

        else:
            self.direction = 0
            angle_diff = 0

        return angle_diff

    def get_angular_velocity(self):

        ### P controller
        
        angle_diff = self.get_turn_direction_and_angle_diff()

        speed = round(self.Kp_angle * angle_diff, 1)

        if speed > 0:
            speed = min(self.pos_top_angular_speed, speed)
        
        else:
            speed = max(self.neg_top_angular_speed, speed)

        self.angular_z_vel = float(self.direction * speed)
        self.get_logger().info(f'angular velocity: {self.angular_z_vel}')	

    def get_linear_velocity(self):

        ### P controller

        noise = 0.01

        diff = self.distance - self.target_distance # positive if too close, negaitve if too far

        speed = round(self.Kp_dist * diff, 2)

        if speed > 0:
            speed = min(self.pos_top_speed, speed)
        
        else:
            speed = max(self.neg_top_speed, speed)

        if diff > noise or diff < -noise:
            self.linear_x_vel = speed
        
        # if diff > noise:
        #     self.linear_x_vel = speed

        # elif diff < -noise:
        #     self.linear_x_vel = speed

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
            vel = Twist()
            vel.linear.x = 0.0
            vel.angular.z = 0.0
            velocity_generator.velocity_publisher.publish(vel)
            break

	#Clean up and shutdown.

    velocity_generator.destroy_node()  
    rclpy.shutdown()


if __name__ == '__main__':
	main()