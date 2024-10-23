# Yi Lian
# Evan Rosenthal

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Twist
from std_msgs.msg import Float32, Float32MultiArray

# etc
import numpy as np
import math

from sensor_msgs.msg import LaserScan


class ObstacleDetection(Node):
    def __init__(self):
        super().__init__('obstacle_detection')
        self.globalPos = Point()
        self.globalAng = 0.0
        self.globalAng_deg = 0.0

        self.ranges = [0.0]
        self.range_min = 0.0
        self.range_max = 0.0

        self.angle_min = 0.0
        self.angle_max = 0.0
        self.angle_increment = 0.0

        self.obstacle_distance = 0.35 
        self.front_angles_limits = [180, 270] # 0-180 deg + 270-360 deg -> robot has 270 deg FOV
        self.obstacle_dected, self.min_range_angle, self.min_range = 0.0, 0.0, 0.0


        #Set up QoS Profiles for passing lidar data over WiFi
        lidar_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

        #Set up QoS Profiles for passing numbers over WiFi
        num_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.RELIABLE,
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
            lidar_qos_profile)
        
        # create object distance and angle publisher
        self.obstacle_range_robot_pose_publisher = self.create_publisher(
				Float32MultiArray, 
				'/obstacle_robot_info/obstacle_dist_robot_pose',
				num_qos_profile)
        self.obstacle_range_robot_pose_publisher

    def global_position_callback(self, msg):
        # self.get_logger().info('receiving robot pose')
        self.globalPos.x, self.globalPos.y, self.globalAng, self.globalAng_deg = msg.data
        # self.get_logger().info('received global pose is x:{}, y:{}, a:{}, a_deg'.format(self.globalPos.x,self.globalPos.y,self.globalAng, self.globalAng_deg))

    def lidar_callback(self, msg):
        self.ranges = msg.ranges
        self.range_min = msg.range_min
        self.range_max = msg.range_max

        # self.get_logger().info(f'received ranges (before preprocessing): {self.ranges}')

        for dist in self.ranges:
            if (dist < self.range_min) or (dist > self.range_max) or (math.isnan(dist)):
                dist = math.inf
  
        # self.get_logger().info(f'received ranges (after preprocessing): {self.ranges}')

        self.angle_min = msg.angle_min
        self.angle_max = msg.angle_max
        self.angle_increment = msg.angle_increment

        # self.get_logger().info(f'lidar angles (min, max, inc): {self.angle_min}, {self.angle_max}, {self.angle_increment}')

        self.obstacle_dected, self.min_range_angle, self.min_range = self.detect_obstacle()

    def calculate_distance(self, angle_deg):

        increment = self.angle_increment
        angle_rad = angle_deg * math.pi / 180
        # self.get_logger().info(f'angle: {angle_deg} deg, {angle_rad} rad')
        # self.get_logger().info(f'angle: {angle_rad} rad')
        distance_idx = int(angle_rad / increment)
        # self.get_logger().info(f'distance index: {distance_idx}')

        if 0 <= distance_idx <= (len(self.ranges)-1): # angle is within lidar measuring range
            distance = self.ranges[distance_idx]

            if not math.isnan(distance): # distance is not nan
                # self.get_logger().info(f'distance: {distance}')
                return distance
            
            else: # distance is nan NEED TO CHECK WHY
                distance = math.inf
                return distance
        else: # angle is not within lidar measuring range

            distance = math.inf
            return distance
        
    def detect_obstacle(self):

        front_angles = []
        front_ranges = []
        obstacle_detected = 0.0 # false

        for angle_deg in range(0, 361):
            if angle_deg <= self.front_angles_limits[0] or angle_deg >= self.front_angles_limits[1]:

                distance = self.calculate_distance(angle_deg)
                front_angles.append(angle_deg)
                front_ranges.append(distance)

        min_range = min(front_ranges)
        min_range_idx = front_ranges.index(min_range)
        min_range_angle = front_angles[min_range_idx]

        if min_range < self.obstacle_distance:
            obstacle_detected = 1.0 # true

        return obstacle_detected, min_range_angle, min_range
    
    def publish_obstacle_range_robot_pose(self):

        msg = Float32MultiArray()
        pose = [float(self.globalPos.x), float(self.globalPos.y), self.globalAng, self.globalAng_deg, float(self.obstacle_dected), float(self.min_range_angle), float(self.min_range)]
        msg.data = pose
        self.obstacle_range_robot_pose_publisher.publish(msg)

        self.get_logger().info('global pose and obstacle info (x, y, ang_rad, ang_deg, obstacle_detected, min_range_angle, min_range): "%s"' % msg.data)

        
    # def average_distance(self):

    #     if self.angles is not None:

    #         distance_list = []
    #         for angle in self.angles:
    #             dist = self.calculate_distance(angle)
    #             if dist is not None:
    #                 distance_list.append(dist)

    #         if len(distance_list) >= 1:
    #             self.distance = sum(distance_list) / len(distance_list)

    #         else: 
    #             self.distance = None

    # def send_distance_and_angle(self):

    #     if self.distance is not None:
    #         self.angle = self.angles[1]
    #         if self.angle == -math.inf:
    #             self.angle = 0.0

    #         msg = Float32MultiArray()
    #         msg.data = [self.distance, self.angle]
    #         self.obstacle_range_robot_pose_publisher(msg)
    #         self.get_logger().info(f'distance (m): {self.distance}, angle (deg): {self.angle}')

def main(args=None):
    rclpy.init(args=args)
    obstacle_detection = ObstacleDetection()

    while rclpy.ok():

        try:
            rclpy.spin_once(obstacle_detection)

            obstacle_detection.publish_obstacle_range_robot_pose()

        except KeyboardInterrupt:
            break
    
    obstacle_detection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()