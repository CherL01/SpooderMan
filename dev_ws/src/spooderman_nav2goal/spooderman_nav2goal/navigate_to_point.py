# Yi Lian
# Evan Rosenthal

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import sys

import numpy as np

from std_msgs.msg import Int64, Float32, Float32MultiArray
from geometry_msgs.msg import PointStamped, PoseStamped, Point, PoseWithCovarianceStamped
from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage

### https://docs.nav2.org/configuration/packages/configuring-regulated-pp.html

class PositionGenerator(Node):

    def __init__(self):		
		# Creates the node.
        super().__init__('position_generator')

        self.map_name = 'map'

        self.point_queue = []

        self.mapPosition = Point()

        self.point_number = 0

        #Set up QoS Profiles for passing numbers over WiFi
        num_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

        #Set up QoS Profiles for passing pose info over WiFi
        pos_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.RELIABLE,
		    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
		    depth=1
		)

        self.map_position_subscriber = self.create_subscription(
				PoseWithCovarianceStamped,
				'/amcl_pose',
                self.map_position_callback,
				pos_qos_profile)
        self.map_position_subscriber # Prevents unused variable warning.

        if len(self.point_queue) < 3:

            self.clicked_point_subscriber = self.create_subscription(
                    PointStamped,
                    '/clicked_point',
                    self.clicked_point_callback,
                    num_qos_profile)
            self.clicked_point_subscriber # Prevents unused variable warning.

        self.position_publisher = self.create_publisher(
				PoseStamped, 
				'/goal_pose',
				num_qos_profile)
        self.position_publisher

    def clicked_point_callback(self, msg):
        x, y, z = msg.point.x, msg.point.y, msg.point.z
        self.get_logger().info(f'received point: {x}, {y}, {z}')

        self.point_queue.append((x, y, z))
        self.get_logger().info(f'point queue: {self.point_queue}')

    def map_position_callback(self, msg):
        self.get_logger().info('receiving robot pose')
        self.mapPosition.x, self.mapPosition.y = msg.pose.pose.position.x, msg.pose.pose.position.y
        self.get_logger().info('received map pose is x:{}, y:{}'.format(self.mapPosition.x, self.mapPosition.y))

    def send_goal(self):

        if len(self.point_queue) >= 3:

            x, y, z = self.point_queue[self.point_number]

            if self.check_goal_reached(x, y):

                self.point_number += 1

                if self.point_number < 3:

                    new_x, new_y, new_z = self.point_queue[self.point_number]
                    self.publish_position(new_x, new_y, new_z)
            
            else:
                self.publish_position(x, y, z)

    def check_goal_reached(self, x, y):

        goal_dist_tolerance = 0.1
        
        if (abs(self.mapPosition.x - x) < goal_dist_tolerance) and (abs(self.mapPosition.y - y) < goal_dist_tolerance):

            self.get_logger().info(f'goal reached: {self.mapPosition.x}, {self.mapPosition.y}')

            return True

        self.get_logger().info(f'goal not reached: {self.mapPosition.x}, {self.mapPosition.y}')

        return False

    def publish_position(self, x, y, z):

        pose = PoseStamped()

        pose.header.frame_id = self.map_name
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        self.get_logger().info(f'publishing goal: {pose.pose.position.x}, {pose.pose.position.y}, {pose.pose.position.z}')

        self.position_publisher.publish(pose)

def main():
    rclpy.init() 
    position_generator = PositionGenerator() 
    
    while rclpy.ok():

        try:

            rclpy.spin_once(position_generator)
            position_generator.send_goal()

            if position_generator.point_number == 3:
                print("All points reached. Shutting down.")
                break

        except KeyboardInterrupt:
            print("Keyboard Interrupt. Shutting down.")
            break

    position_generator.destroy_node()  
    rclpy.shutdown()


if __name__ == '__main__':
	main()