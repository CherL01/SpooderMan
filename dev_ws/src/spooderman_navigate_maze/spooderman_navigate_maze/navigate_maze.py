# Yi Lian
# Evan Rosenthal

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point, Quaternion, Twist
from geometry_msgs.msg import PointStamped, PoseStamped, Point, PoseWithCovarianceStamped
from std_msgs.msg import Float32, Float32MultiArray
from tensorflow.keras.models import load_model

# etc
import numpy as np
import math
import time
import cv2
from cv_bridge import CvBridge
import os

# imports from other files
from classifier import load_data, evaluate_model

class NavigateMaze(Node):

    def __init__(self):		

        super().__init__('navigate_maze')

        # maze setup
        self.maze = [[f'{row}_{col}' for col in range(6)] for row in ['A', 'B', 'C']]
        self.maze_coords = { # TODO: set up coords for each square in maze, can find from click point in rviz
            'A_0',
            'A_1', 
            'A_2', 
            'A_3', 
            'A_4', 
            'A_5', 
            'B_0', 
            'B_1', 
            'B_2', 
            'B_3', 
            'B_4', 
            'B_5', 
            'C_0', 
            'C_1', 
            'C_2', 
            'C_3', 
            'C_4', 
            'C_5'
            }
        self.directions = ['N', 'E', 'S', 'W']
        self.direction_quaternion = { # TODO: set up quaternion directions for each direction, can i calculate based on current quaternion?
            'N', 
            'E', 
            'S', 
            'W'
            }
        self.walls = {
            'N': {0: ['A'], 1: ['C', 'A'], 2: ['B', 'A'], 3: ['A'], 4: ['B', 'A'], 5: ['A']}, # 0, 1, 2, 3, 4, 5
            'E': {'A': [3, 5], 'B': [0, 2, 3, 4, 5], 'C': [1, 5]}, # A, B, C
            'S': {0: ['C'], 1: ['B', 'C'], 2: ['A', 'C'], 3: ['C'], 4: ['A', 'C'], 5: ['C']}, # 0, 1, 2, 3, 4, 5
            'W': {'A': [4, 0], 'B': [5, 4, 3, 1, 0], 'C': [2, 0]} # A, B, C
        }
        
        # robot setup
        self.state = 0 # initialized as 0 to start classification
        self.action = None # initialized as None because have not classified direction sign yet
        self.waypoint_reached = True # initialized as True to start classification
        self.classification_result = None # initialized as None to start classification
        self.goal_pose = None
        self.maze_position = None
        self.current_direction = None

        ### qos profiles
        # Set up QoS Profiles for passing numbers over WiFi
        num_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

        # Set up QoS Profiles for passing pose info over WiFi
        pos_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.RELIABLE,
		    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
		    depth=1
		)
        
        # Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)
        ###

        ### navigation setup
        self.map_name = 'map'
        self.mapPosition = PoseStamped()

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
        
        ### classification setup
        self.frame_count = 0
        self.max_frame_count = 15
        self.save_folder = os.path.abspath(os.path.join("dev_ws", "src", "spooderman_navigate_maze", "maze_images"))
        self.model_path = os.path.abspath(os.path.join("dev_ws", "src", "spooderman_navigate_maze", "spooderman_navigate_maze", "classifier_model.h5"))
        self.test_folder = self.save_folder
        
        #Declare that the minimal_video_subscriber node is subcribing to the /camera/image/compressed topic.
        self._video_subscriber = self.create_subscription(
				CompressedImage,
				'/image_raw/compressed',
				self._image_callback,
				image_qos_profile)
        self._video_subscriber
        ###

    def map_position_callback(self, msg):
        self.get_logger().info('receiving robot pose')
        self.mapPosition = msg
        self.get_logger().info(f'received map pose is x:{self.mapPosition.pose.position.x}, y:{self.mapPosition.pose.position.y}')
        
    def _image_callback(self, CompressedImage):	
		# The "CompressedImage" is transformed to a color image in BGR space and is store in "_imgBGR"
        self._imgBGR = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")

    def publish_position(self, x, y, z, qx, qy, qz, qw):

        # TODO: modify to accomodate for quaternion

        pose = PoseStamped()

        pose.header.frame_id = self.map_name
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        # placeholder quaternion
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        
        self.goal_pose = pose

        self.get_logger().info(f'publishing goal: {pose.pose.position.x}, {pose.pose.position.y}, {pose.pose.orientation.x}, {pose.pose.orientation.y}, {pose.pose.orientation.z}, {pose.pose.orientation.w}')

        self.position_publisher.publish(pose)
        
    def localize(self):
        
        # TODO: add code to localize robot at start of maze
        # localize based on 2d pose estimate in rviz
        # find closest square in maze to robot's current position and set as current position
        # find closest quaternion direction to robot's current quaternion and set as current direction
        
        # placeholder
        self.maze_position = 'C_1'
        self.current_direction = 'N'

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

            return True

        return False
    
    def take_picture(self):
        
        frame_path = os.path.join(self.save_folder, f"frame_{self.frame_count}.png")
        cv2.imwrite(frame_path, self._imgBGR)
        
    def delete_pictures(self):
            
        for i in range(self.max_frame_count):
            frame_path = os.path.join(self.save_folder, f"frame_{i}.png")
            os.remove(frame_path)
    
    def classify_direction_sign(self):

        # call classifier here
        # get average classification result

        # TODO: add code to take picture and classify direction sign
        if self.frame_count < self.max_frame_count: # take pictures for classification
            self.take_picture()
            self.frame_count += 1
            
        else: # got enough pictures, start classification
            
            self.logger.get_info('starting classification, loading model')
            model = load_model(self.model_path)
            
            predicted_labels = evaluate_model(model, self.test_folder)
            self.logger.get_info(f'predicted labels: {predicted_labels.shape}') # need to check shape
            
            self.classification_result = np.argmax(np.mean(predicted_labels, axis=0))
            self.logger.get_info(f'classification result: {self.classification_result}')
            
            self.delete_pictures()
            self.frame_count = 0

    def navigate_to_waypoint(self):
            
        # follow direction sign
        if self.action == 1:
            self.move('left')

        elif self.action == 2:
            self.move('right')

        elif self.action == 3 or self.action == 4:
            self.move('u-turn')

    def move(self, turn_direction):

        # TODO: add code to calculate coordinate + quaternion for turning left, and send to nav2 stack (can use lab 5 code)

        new_heading = self.get_new_heading(self.current_direction, turn_direction)
        next_square = self.get_next_square(self.maze_position, new_heading)
        
        x, y, z = self.maze_coords[next_square]
        qx, qy, qz, qw = self.direction_quaternion[new_heading]
        
        self.publish_position(x, y, z, qx, qy, qz, qw)
    
    def get_new_heading(self, current_heading, turn_direction):
        
        current_heading_index = self.directions.index(current_heading)
        if turn_direction == 'left':
            new_heading_index = (current_heading_index - 1) % 4
            
        elif turn_direction == 'right':
            new_heading_index = (current_heading_index + 1) % 4
            
        elif turn_direction == 'u-turn':
            new_heading_index = (current_heading_index + 2) % 4
            
        else:
            new_heading_index = 0
            self.logger.get_info('invalid turn direction')
            
        new_heading = self.directions[new_heading_index]
        
        return new_heading
    
    def get_next_square(self, current_square, new_heading):
        
        row = current_square.split('_')[0] # A, B, C
        col = int(current_square.split('_')[1]) # 0, 1, 2, 3, 4, 5
        
        if new_heading == 'E' or new_heading == 'W': # look at cols, row stays the same
            if new_heading == 'E': # E is increasing col, check when col > current col
                walls_list = self.walls['E'][col]
                for wall in walls_list:
                    if wall > col:
                        next_square = f'{row}_{wall}'
                        break
                    
            else:
                walls_list = self.walls['W'][row]
                for wall in walls_list:
                    if wall < col:
                        next_square = f'{row}_{wall}'
                        break
            
        elif new_heading == 'N' or new_heading == 'S': # look at rows, col stays the same
            if new_heading == 'N': # N is decreasing row, check when row < current row
                walls_list = self.walls['N'][row]
                for wall in walls_list:
                    if wall < row:
                        next_square = f'{wall}_{col}'
                        break
                
            else:
                walls_list = self.walls['S'][row]
                for wall in walls_list:
                    if wall > row:
                        next_square = f'{wall}_{col}'
                        break
                    
        else:
            self.logger.get_info('invalid heading')
            
        return next_square
    
    def compute_angle_difference(self, q1, q2):
        
        # Compute the relative quaternion
        q1_conjugate = np.array([-q1[0], -q1[1], -q1[2], q1[3]])
        relative = np.array([
            q1_conjugate[3]*q2[0] + q1_conjugate[0]*q2[3] + q1_conjugate[1]*q2[2] - q1_conjugate[2]*q2[1],
            q1_conjugate[3]*q2[1] - q1_conjugate[0]*q2[2] + q1_conjugate[1]*q2[3] + q1_conjugate[2]*q2[0],
            q1_conjugate[3]*q2[2] + q1_conjugate[0]*q2[1] - q1_conjugate[1]*q2[0] + q1_conjugate[2]*q2[3],
            q1_conjugate[3]*q2[3] - q1_conjugate[0]*q2[0] - q1_conjugate[1]*q2[1] - q1_conjugate[2]*q2[2]
        ])
        relative = relative / np.linalg.norm(relative)

        # Compute the angle of rotation
        angle_radians = 2 * np.arccos(relative[3])
        return abs(np.rad2deg(angle_radians))

    def check_waypoint_reached(self):

        # TODO: add code to check if waypoint reached (can use lab 5 code but need to modify quaternion comparison)

        x, y, z, qx, qy, qz, qw = self.goal_pose.pose.position.x, self.goal_pose.pose.position.y, self.goal_pose.pose.position.z, self.goal_pose.pose.orientation.x, self.goal_pose.pose.orientation.y, self.goal_pose.pose.orientation.z, self.goal_pose.pose.orientation.w
        x_map, y_map, z_map, qx_map, qy_map, qz_map, qw_map = self.mapPosition.pose.position.x, self.mapPosition.pose.position.y, self.mapPosition.pose.position.z, self.mapPosition.pose.orientation.x, self.mapPosition.pose.orientation.y, self.mapPosition.pose.orientation.z, self.mapPosition.pose.orientation.w

        angle_diff_deg = self.compute_angle_difference(np.array([qx, qy, qz, qw]), np.array([qx_map, qy_map, qz_map, qw_map]))

        goal_dist_tolerance = 0.08
        goal_angle_tolerance = 5
        
        if (abs(x_map - x) < goal_dist_tolerance) and (abs(y_map - y) < goal_dist_tolerance) and (angle_diff_deg < goal_angle_tolerance):

            self.get_logger().info('waypoint reached')

            return True

        self.get_logger().info('still travelling to waypoint')

        return False

def main():
    rclpy.init()
    navigate_maze = NavigateMaze()
    localize = True
    
    while rclpy.ok():

        try:
            
            rclpy.spin_once(navigate_maze)
            
            if localize:
                navigate_maze.localize()
                localize = False
            
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