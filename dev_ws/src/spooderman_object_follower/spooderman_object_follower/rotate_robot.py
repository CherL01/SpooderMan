import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import sys

import numpy as np

from std_msgs.msg import Int64, Float32
from geometry_msgs.msg import Twist

class VelocityGenerator(Node):

    def __init__(self):		
		# Creates the node.
        super().__init__('velocity_generator')
		
		#Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
		    history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
		    durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE,
		    depth=1
		)

		# Declare that the velocity_generator node is subcribing to the /object_detect/coords topic.
        self.coordinate_subscriber = self.create_subscription(
				Int64,
				'/object_detect/coords',
                self.coords_callback,
				image_qos_profile)
        self.coordinate_subscriber # Prevents unused variable warning.

        # create velocity publisher
        self.velocity_publisher = self.create_publisher(
				Twist, 
				'/cmd_vel',
				image_qos_profile)
        self.velocity_publisher

        self.x_coord = None

    def coords_callback(self, msg):
        self.x_coord = msg.data

    def get_spin_direction(self):

        left = 0
        right = 255
        center = (left + right) // 2

        if self.x_coord < center:
            self.direction = 1
        
        elif self.x_coord > center:
            self.direction = -1

        else:
            self.direction = 0

    def get_spin_velocity(self):
        self.get_spin_direction()
        speed = 2
        self.vel = float(self.direction * speed)

    def publish_spin_velocity(self):
        if self.x_coord != None:
            self.get_spin_velocity()
            vel = Twist()
            vel.angular.z = self.vel
            self.velocity_publisher.publish(vel)
            self.get_logger().info('angular velocity: "%s"' % vel.angular.z)	

    def get_user_input(self):
        return self._user_input

def main():
    rclpy.init() #init routine needed for ROS2.
    velocity_generator = VelocityGenerator() #Create class object to be used.
    
    while rclpy.ok():
        rclpy.spin_once(velocity_generator) # Trigger callback processing.
        velocity_generator.publish_spin_velocity()

	#Clean up and shutdown.
    velocity_generator.destroy_node()  
    rclpy.shutdown()


if __name__ == '__main__':
	main()