# Yi Lian
# Evan Rosenthal

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Twist
from geometry_msgs.msg import PointStamped, PoseStamped, Point, PoseWithCovarianceStamped
from std_msgs.msg import Float32, Float32MultiArray

# etc
import numpy as np
import math
import time

class NavigateMaze(Node):

    def __init__(self):		

        super().__init__('navigate_maze')

        # maze setup
        self.maze = [[f'{row}{col}' for col in range(6)] for row in ['A', 'B', 'C']]
        self.maze_coords = {} # TODO: set up coords for each square in maze
        self.directions = ['N', 'E', 'S', 'W']
        self.direction_quaternion = { # TODO: set up quaternion directions for each direction, can i calculate based on current quaternion?
            'N': 0, 
            'E': 1, 
            'S': 2, 
            'W': 3
            }
        
        # robot setup
        self.state = 0 # initialized as 0 to start classification
        self.action = None # initialized as None because have not classified direction sign yet
        self.waypoint_reached = True # initialized as True to start classification
        self.classification_result = None # initialized as None to start classification
        self.waypoint_coords = None # initialized as None to start classification
        self.currect_direction = 'N' # initialized as 'N' (ALWAYS FACE THE ROBOT NORTH AT THE START)

        ### qos profiles
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

        ### navigation setup
        self.map_name = 'map'
        self.mapPosition = Point()

        self.map_position_subscriber = self.create_subscription(
				PoseWithCovarianceStamped,
				'/amcl_pose',
                self.map_position_callback,
				pos_qos_profile)
        self.map_position_subscriber

        self.position_publisher = self.create_publisher(
				PoseStamped, 
				'/goal_pose',
				num_qos_profile)
        self.position_publisher
        ###

    def map_position_callback(self, msg):
        self.get_logger().info('receiving robot pose')
        self.mapPosition.x, self.mapPosition.y, self.current_quaternion = msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.orientation
        self.get_logger().info('received map pose is x:{}, y:{}, quaternion:{}'.format(self.mapPosition.x, self.mapPosition.y, self.current_quaternion))

    def publish_position(self, x, y, quaternion):

        # TODO: modify to accomodate for quaternion

        pose = PoseStamped()

        pose.header.frame_id = self.map_name
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        # placeholder quaternion
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        self.get_logger().info(f'publishing goal: {pose.pose.position.x}, {pose.pose.position.y}, {pose.pose.orientation.x}, {pose.pose.orientation.y}, {pose.pose.orientation.z}, {pose.pose.orientation.w}')

        self.position_publisher.publish(pose)

    def get_state(self):
        
        ### classification classes:
        # 0: nothing/other/ travelling
        # 1: left turn?
        # 2: right turn?
        # 3: u-turn?
        # 4: stop sign? (also u-turn)
        # 5: goal

        ### robot states:
        # 0: classifying direction sign
        # 1: travel to next waypoint
        # 2: goal reached

        ### obstacle detection should be accounted for in nav2 stack -> do not need to worry about it here?
        ### shouldn't be doing classification during travel anyway -> classification class 0 not too useful (unless nav2 stack fails)

        if self.waypoint_reached is True:

            # did not classify direction sign yet
            if self.classification_result is None:
                
                self.state = 0

            # classified direction sign
            else: 

                # goal reached
                if self.classification_result == 5:
                    self.state = 2

                # did not reach goal yet, follow direction sign
                else:
                    self.state = 1
                    self.action = self.classification_result
                    self.waypoint_reached = False
                    self.classification_result = None
                    self.navigate_to_waypoint()

        if self.state == 0:
            self.get_logger().info(f'classifying direction sign')
            self.classify_direction_sign()

        elif self.state == 1:
            self.get_logger().info(f'traveling to next waypoint')
            self.waypoint_reached = self.check_waypoint_reached()

        elif self.state == 2:
            self.get_logger().info(f'goal reached!')
            self.stop()

            return True

        return False
    
    def classify_direction_sign(self):

        # call classifier here
        # get average classification result

        # TODO: add code to take picture and classify direction sign

        # placeholder
        self.classification_result = 1

    def navigate_to_waypoint(self):
            
        # follow direction sign
        if self.action == 1:
            self.turn_left()

        elif self.action == 2:
            self.turn_right()

        elif self.action == 3:
            self.u_turn()

        elif self.action == 4:
            self.u_turn()

        # follow path
        else:
            self.follow_path()

    def turn_left(self):

        # TODO: add code to calculate coordinate + quaternion for turning left, and send to nav2 stack (can use lab 5 code)

        pass

    def turn_right(self):
        pass

    def u_turn(self):
        pass

    def follow_path(self):
        pass

    def check_waypoint_reached(self):

        # TODO: add code to check if waypoint reached (can use lab 5 code but need to modify quaternion comparison)

        x, y, z = self.waypoint_coords

        goal_dist_tolerance = 0.1
        
        if (abs(self.mapPosition.x - x) < goal_dist_tolerance) and (abs(self.mapPosition.y - y) < goal_dist_tolerance):

            self.get_logger().info(f'goal reached: {self.mapPosition.x}, {self.mapPosition.y}')

            return True

        self.get_logger().info(f'goal not reached: {self.mapPosition.x}, {self.mapPosition.y}')

        return False

def main():
    rclpy.init()
    navigate_maze = NavigateMaze()
    
    while rclpy.ok():

        try:
            rclpy.spin_once(navigate_maze) 
            navigation_complete = navigate_maze.get_state()

            if navigation_complete is True:
                break

        except KeyboardInterrupt:
            break

	#Clean up and shutdown.

    navigate_maze.destroy_node()  
    rclpy.shutdown()


if __name__ == '__main__':
	main()