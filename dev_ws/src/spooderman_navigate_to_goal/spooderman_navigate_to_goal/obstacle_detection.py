import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Int64, Float32, Float32MultiArray

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np

import math

class ObstacleDetection(Node):
    def __init__(self):
        super().__init__('obstacle_detection')

        #Set up QoS Profiles for passing numbers over WiFi
        num_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

        self.global_position_subscriber = self.create_subscription(
				Float32MultiArray,
				'/robot_position/global_pose',
                self.global_position_callback,
				num_qos_profile)
        self.global_position_subscriber # Prevents unused variable warning.

        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            num_qos_profile)
        
        self.x_coords = None
        self.y_coords = None
        self.angles = None

        self.ranges = None
        self.range_min = 0.0
        self.range_max = 0.0

        self.angle_min = None
        self.angle_max = None
        self.angle_increment = None

        self.angle = None
        self.distance = None

        self.target_distance = 0.35 # meters

        # create object distance and angle publisher
        self.obstacle_range_robot_pose_publisher = self.create_publisher(
				Float32MultiArray, 
				'/obstacle_robot_info/obstacle_dist_robot_pose',
				num_qos_profile)
        self.obstacle_range_robot_pose_publisher

    def lidar_callback(self, msg):
        self.ranges = msg.ranges
        self.range_min = msg.range_min
        self.range_max = msg.range_max

        # self.get_logger().info(f'received ranges (before preprocessing): {self.ranges}')

        for dist in self.ranges:
            if (dist < self.range_min) or (dist > self.range_max) or (math.isnan(dist)):
                dist = math.inf
  
        self.get_logger().info(f'received ranges (after preprocessing): {self.ranges}')

        self.angle_min = msg.angle_min
        self.angle_max = msg.angle_max
        self.angle_increment = msg.angle_increment

        self.get_logger().info(f'lidar angles (min, max, inc): {self.angle_min}, {self.angle_max}, {self.angle_increment}')

    def calculate_distance(self, angle):

        if angle != -math.inf: # angle is not -inf = object in frame # NEED TO FIX NONE

            increment = self.angle_increment
            angle_rad = angle * math.pi / 180
            self.get_logger().info(f'angle: {angle} deg, {angle_rad} rad')
            distance_idx = int(angle_rad / increment)
            self.get_logger().info(f'distance index: {distance_idx}')

            if 0 <= distance_idx <= (len(self.ranges)-1): # angle is within lidar measuring range
                distance = self.ranges[distance_idx]

                if not math.isnan(distance): # distance is not nan
                    self.get_logger().info(f'distance: {distance}')
                    return distance
                
                else: # distance is nan NEED TO CHECK WHY
                    distance = None
                    return distance
            else: # angle is not within lidar measuring range

                distance = None
                return distance
            
        else: # object is not in frame, want robot to remain neutral

            distance = self.target_distance

            return distance
        
    def average_distance(self):

        if self.angles is not None:

            distance_list = []
            for angle in self.angles:
                dist = self.calculate_distance(angle)
                if dist is not None:
                    distance_list.append(dist)

            if len(distance_list) >= 1:
                self.distance = sum(distance_list) / len(distance_list)

            else: 
                self.distance = None

    def send_distance_and_angle(self):

        if self.distance is not None:
            self.angle = self.angles[1]
            if self.angle == -math.inf:
                self.angle = 0.0

            msg = Float32MultiArray()
            msg.data = [self.distance, self.angle]
            self.obstacle_range_robot_pose_publisher(msg)
            self.get_logger().info(f'distance (m): {self.distance}, angle (deg): {self.angle}')

def main(args=None):
    rclpy.init(args=args)
    object_dist = ObstacleDetection()

    while rclpy.ok():

        try:
            rclpy.spin_once(object_dist)

            if object_dist.ranges is not None:

                object_dist.average_distance()
                object_dist.send_distance_and_angle()

        except KeyboardInterrupt:
            break
    
    object_dist.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()