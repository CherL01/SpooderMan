import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Int64, Float32, Float32MultiArray

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import math

class GetObjectRange(Node):
    def __init__(self):
        super().__init__('object_chaser')

        #Set up QoS Profiles for passing numbers over WiFi
        num_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

        # Declare that the velocity_generator node is subcribing to the /object_detect/coords topic.
        self.coordinate_subscriber = self.create_subscription(
				Float32,
				'/object_detect/x_coord',
                self.coords_callback,
				num_qos_profile)
        self.coordinate_subscriber # Prevents unused variable warning.

        # Declare that the velocity_generator node is subcribing to the /object_detect/coords topic.
        self.angle_subscriber = self.create_subscription(
				Float32,
				'/object_detect/angle',
                self.angle_callback,
				num_qos_profile)
        self.angle_subscriber # Prevents unused variable warning.

        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            num_qos_profile)
        
        self.x_coord = None
        self.angle = None
        self.ranges = None
        self.angles = None
        self.distance = None

        # create object distance and angle publisher
        self.dist_and_angle_publisher = self.create_publisher(
				Float32MultiArray, 
				'/object_detect/dist_and_angle',
				num_qos_profile)
        self.dist_and_angle_publisher

    def coords_callback(self, msg):
        self.x_coord = msg.data
        self.get_logger().info('received center (x-coord): "%s"' % msg.data)

    def angle_callback(self, msg):
        self.angle = msg.data
        self.get_logger().info('received center (angle in deg): "%s"' % msg.data)

    def lidar_callback(self, msg):
        self.ranges = msg.ranges
        self.angles = [msg.angle_min, msg.angle_max, msg.angle_increment]
        # min_distance = min(self.ranges)
        # self.get_logger().info('lidar ranges: "%s"' %msg.ranges)
        # self.get_logger().info('lidar min distance: "%s"' %min_distance)
        self.get_logger().info(f'lidar angles: "{self.angles}')
        self.calculate_distance()

    def calculate_distance(self):
        increment = self.angles[2]
        angle_rad = self.angle * math.pi / 180
        self.get_logger().info(f'angle (rad): {angle_rad}')
        distance_idx = int(angle_rad / increment)
        self.get_logger().info(f'distance index: {distance_idx}')

        if 0 <= distance_idx <= (len(self.ranges)-1):
            self.distance = self.ranges[distance_idx]
            self.get_logger().info(f'distance: {self.distance}')

        else:
            self.distance = None
        

    def send_distance_and_angle(self):

        if self.distance != None:
            msg = Float32MultiArray()
            msg.data = [self.distance, self.angle]
            self.dist_and_angle_publisher.publish(msg)
            self.get_logger().info(f'distance: {self.distance}, angle: {self.angle}')

def main(args=None):
    rclpy.init(args=args)
    object_dist = GetObjectRange()

    while rclpy.ok():

        try:
            rclpy.spin_once(object_dist)

            # calculate distance
            object_dist.send_distance_and_angle()


        except KeyboardInterrupt:
            break
    
    object_dist.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()