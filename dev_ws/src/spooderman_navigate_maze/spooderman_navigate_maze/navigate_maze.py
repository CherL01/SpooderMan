# Yi Lian
# Evan Rosenthal

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Point, Quaternion, Twist
from geometry_msgs.msg import PointStamped, PoseStamped, Point, PoseWithCovarianceStamped
from std_msgs.msg import Float32, Float32MultiArray
from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage
from tensorflow.keras.models import load_model
import joblib

# etc
import numpy as np
import math
import time
import cv2
from cv_bridge import CvBridge
import os

# imports from other files
from .cnn import evaluate_model
from .svm import evaluate_svm


class NavigateMaze(Node):

    def __init__(self):		

        super().__init__('navigate_maze')

        # maze setup
        self.maze = [[f'{row}_{col}' for col in range(6)] for row in ['A', 'B', 'C']]
        self.maze_coords = { # TODO: set up coords for each square in maze, can find from click point in rviz
            'A_0': (-0.9, 1.81, 0),
            'A_1': (-0.18, 1.81, 0), 
            'A_2': (0.79, 1.81, 0), 
            'A_3': (1.70, 1.81, 0), 
            'A_4': (2.73, 1.81, 0), 
            'A_5': (3.53, 1.81, 0), 
            'B_0': (-0.9, 0.89, 0), 
            'B_1': (-0.18, 0.89, 0), 
            'B_2': (0.79, 0.89, 0), 
            'B_3': (1.70, 0.89, 0), 
            'B_4': (2.73, 0.89, 0), 
            'B_5': (3.53, 0.89, 0), 
            'C_0': (-0.9, 0.02, 0), 
            'C_1': (-0.18, 0.02, 0), 
            'C_2': (0.79, 0.02, 0), 
            'C_3': (1.70, 0.02, 0), 
            'C_4': (2.73, 0.02, 0),
            'C_5': (3.53, 0.02, 0)
            }
        self.directions = ['N', 'E', 'S', 'W']
        self.direction_quaternion = { # TODO: set up quaternion for each direction, can find from /amcl_pose topic
            'N': (0, 0, math.sqrt(2) / 2, math.sqrt(2) / 2), # 90 deg
            'E': (0, 0, 0, 1), # 0 deg
            'S': (0, 0, -math.sqrt(2) / 2, math.sqrt(2) / 2), # -90 deg
            'W': (0, 0, 1, 0) # 180 deg
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
        self.goal_pose_nums = None
        self.maze_position = None
        self.current_direction = None
        self.goal_position = None

        ### TESTING PURPOSES ONLY, COMMENT OUT AFTER ###
        self.test_classification_results = [3, 2, 2, 1, 5]
        self.classification_index = 0

        ### qos profiles
        # Set up QoS Profiles for passing numbers over WiFi
        num_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

        vel_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.RELIABLE,
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
		    reliability=QoSReliabilityPolicy.RELIABLE,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)
        ###

        ### navigation setup
        self.map_name = 'map'
        self.mapPosition = None
        self.distance_remaining = None
        self.estimated_time_remaining = None

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

        self.feedback_subscriber = self.create_subscription(
				NavigateToPose_FeedbackMessage,
				'/navigate_to_pose/_action/feedback',
                self.feedback_callback,
				num_qos_profile)
        self.map_position_subscriber

        self.velocity_publisher = self.create_publisher(
				Twist, 
				'/cmd_vel',
				vel_qos_profile)
        self.velocity_publisher
        ###
        
        ### classification setup
        self.frame_count = 0
        self.max_frame_count = 15
        self.save_folder = os.path.abspath(os.path.join("src", "spooderman_navigate_maze", "maze_images"))
        self.test_folder = self.save_folder
        self.label_to_name = {0: 'nothing', 1: 'left', 2: 'right', 3: 'u-turn', 4: 'u-turn', 5: 'goal'}
        self.model = 'SVM' # 'CNN', 'TESTING'
        
        if self.model == 'CNN':
            self.model_path = os.path.abspath(os.path.join("src", "spooderman_navigate_maze", "spooderman_navigate_maze", "classifier_model.h5"))
        elif self.model == 'SVM':
            self.model_path = os.path.abspath(os.path.join("src", "spooderman_navigate_maze", "spooderman_navigate_maze", "svm_model.pkl"))
        
        self._video_subscriber = self.create_subscription(
				Image,
				'/image_raw',
				self._image_callback,
				image_qos_profile)
        self._video_subscriber
        ###

    def map_position_callback(self, msg):
        self.get_logger().info('receiving robot pose')
        self.mapPosition = msg
        self.get_logger().info(f'received map pose is x:{self.mapPosition.pose.pose.position.x}, y:{self.mapPosition.pose.pose.position.y}')

    def feedback_callback(self, msg):
        self.distance_remaining, self.estimated_time_remaining = msg.feedback.distance_remaining, msg.feedback.estimated_time_remaining.sec
        self.get_logger().info(f'received feedback, distance remaining: {self.distance_remaining}, time remaining: {self.estimated_time_remaining}')
        
    def _image_callback(self, img):	
        self.get_logger().info('receiving image')
        self._imgBGR = CvBridge().imgmsg_to_cv2(img, "bgr8")
        self.get_logger().info('received image')

    def publish_position(self, x, y, z, qx, qy, qz, qw):

        # TODO: modify to accomodate for quaternion

        pose = PoseStamped()

        pose.header.frame_id = self.map_name
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)

        # placeholder quaternion
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)
        
        self.goal_pose = pose

        # self.get_logger().info(f'publishing goal: {pose.pose.position.x}, {pose.pose.position.y}, {pose.pose.position.z}, {pose.pose.orientation.x}, {pose.pose.orientation.y}, {pose.pose.orientation.z}, {pose.pose.orientation.w}')

        self.position_publisher.publish(pose)

    def publish_velocity(self, linear_x_vel=0.0, angular_z_vel=0.0):
        vel = Twist()
        vel.linear.x = linear_x_vel
        vel.angular.z = angular_z_vel
        self.velocity_publisher.publish(vel)
        self.get_logger().info(f'publishing velocity: {vel}')
        
    def localize(self):
        
        # TODO: add code to localize robot at start of maze
        # localize based on 2d pose estimate in rviz
        # find closest square in maze to robot's current position and set as current position
        # find closest quaternion direction to robot's current quaternion and set as current direction OR always start facing N?

        try:
            self.get_logger().info('localizing robot')
            best_square, smallest_dist = None , math.inf
            for square, coord in self.maze_coords.items():
                dist = math.dist([coord[0], coord[1]], [self.mapPosition.pose.pose.position.x, self.mapPosition.pose.pose.position.y])
                if dist < smallest_dist:
                    smallest_dist = dist
                    best_square = square

            best_direction, smallest_angle_diff = None, math.inf
            for direction, quaternion in self.direction_quaternion.items():
                angle_diff = self.compute_angle_difference(np.array([quaternion[0], quaternion[1], quaternion[2], quaternion[3]]), np.array([self.mapPosition.pose.pose.orientation.x, self.mapPosition.pose.pose.orientation.y, self.mapPosition.pose.pose.orientation.z, self.mapPosition.pose.pose.orientation.w]))
                if angle_diff < smallest_angle_diff:
                    best_direction = direction
                    smallest_angle_diff = angle_diff

            self.maze_position = best_square
            self.current_direction = best_direction
            self.get_logger().info(f'current position: {self.maze_position}, current direction: {self.current_direction}')

        except AttributeError:
            self.get_logger().info('no map position received yet, cannot localize robot')
            return True
        
        return False

        # # placeholder for testing purposes, COMMENT OUT AFTER
        # self.maze_position = 'C_1'
        # self.current_direction = 'N'

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
            # self.get_logger().info(f'traveling to next waypoint')
            self.waypoint_reached = self.check_waypoint_reached()

        elif self.state == 2:
            self.get_logger().info(f'goal reached!')
            self.stop()

            return True

        return False
    
    def stop(self):
        linear_x_vel = 0.0
        angular_z_vel = 0.0
        self.publish_velocity(linear_x_vel, angular_z_vel)
        self.get_logger().info('stopping')
        time.sleep(10)
    
    def take_picture(self):
        
        frame_path = os.path.join(self.save_folder, f"frame_{self.frame_count}.png")

        cv2.imwrite(frame_path, self._imgBGR)
        # self.get_logger().info(f'saved frame {frame_path}')
        
    def delete_pictures(self):
            
        for i in range(self.max_frame_count):
            frame_path = os.path.join(self.save_folder, f"frame_{i}.png")
            os.remove(frame_path)
    
    def classify_direction_sign(self):

        # call classifier here
        # get average classification result

        # TODO: add code to take picture and classify direction sign
        try:
            if self.frame_count < self.max_frame_count: # take pictures for classification
                self.take_picture()
                self.frame_count += 1
                
            # else: # got enough pictures, start classification
                
            #     self.logger.get_info('starting classification, loading model')
            #     model = load_model(self.model_path)
                
            #     predicted_labels = evaluate_model(model, self.test_folder)
            #     self.logger.get_info(f'predicted labels: {predicted_labels.shape}') # need to check shape
                
            #     self.classification_result = np.argmax(np.mean(predicted_labels, axis=0))
            #     self.logger.get_info(f'classification result: {self.classification_result}')
                
            #     self.delete_pictures()
            #     self.frame_count = 0

            else:

                if self.model == 'TESTING':
                    ### placeholder for testing purposes, COMMENT OUT AFTER
                    self.classification_result = self.test_classification_results[self.classification_index]
                    self.classification_index += 1
                    ###
                    
                elif self.model == 'CNN':

                    model = load_model(self.model_path)
                    results = evaluate_model(model, self.test_folder)

                    # return class that appears most frequently
                    self.classification_result = np.argmax(np.bincount(results))
                    
                elif self.model == 'SVM':
                    
                    model = joblib.load(self.model_path)
                    results = evaluate_svm(model, self.test_folder)
                    
                    # return class that appears most frequently
                    self.classification_result = np.argmax(np.bincount(results))
                    
                self.get_logger().info(f'classification result: {self.classification_result} - {self.label_to_name[self.classification_result]}')

                self.frame_count = 0

        except AttributeError:
            self.get_logger().info('no image received yet, cannot classify direction sign')

    def navigate_to_waypoint(self):
            
        # follow direction sign
        if self.action == 1:
            self.move('left')

        elif self.action == 2:
            self.move('right')

        elif self.action == 3 or self.action == 4:
            self.move('u-turn')

        elif self.action == 0:
            self.get_logger().info('no wall detected, turning left')
            self.move('left')

    def move(self, turn_direction):

        # TODO: add code to calculate coordinate + quaternion for turning left, and send to nav2 stack (can use lab 5 code)

        new_heading = self.get_new_heading(self.current_direction, turn_direction)
        next_square = self.get_next_square(self.maze_position, new_heading)

        self.get_logger().info(f'moving {next_square} facing {new_heading}')
        self.goal_position = (next_square, new_heading)
        
        x, y, z = self.maze_coords[next_square]
        qx, qy, qz, qw = self.direction_quaternion[new_heading]

        self.goal_pose_nums = (x, y, z, qx, qy, qz, qw)
        
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
        
        if new_heading == 'E' or new_heading == 'W': # want to index the row, compared col number (wall), row stays the same
            if new_heading == 'E': # E is increasing col, check when col > current col
                walls_list = self.walls['E'][row]
                for wall in walls_list:
                    if wall >= col:
                        next_square = f'{row}_{wall}'
                        break
                    
            else:
                walls_list = self.walls['W'][row]
                for wall in walls_list:
                    if wall <= col:
                        next_square = f'{row}_{wall}'
                        break
            
        elif new_heading == 'N' or new_heading == 'S': # want to index the col, compared row number (wall), col stays the same
            if new_heading == 'N': # N is decreasing row, check when row < current row
                walls_list = self.walls['N'][col]
                for wall in walls_list:
                    if wall <= row:
                        next_square = f'{wall}_{col}'
                        break
                
            else:
                walls_list = self.walls['S'][col]
                for wall in walls_list:
                    if wall >= row:
                        next_square = f'{wall}_{col}'
                        break
                    
        else:
            self.logger.get_info('invalid heading')
            
        return next_square
    
    def compute_angle_difference(self, q1, q2):

        # TODO: FIX ANGLE DIFFERENCE CALCULATION!!!
        
       # Normalize the quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Compute the dot product
        dot_product = np.dot(q1, q2)
        
        # Clamp the dot product to the range [-1, 1] to avoid numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Compute the angle in radians
        angle_radians = 2 * np.arccos(np.abs(dot_product))
        
        # Convert the angle to degrees (optional)
        angle_degrees = np.degrees(angle_radians)
        
        return angle_degrees

    def check_waypoint_reached(self):

        # TODO: add code to check if waypoint reached (can use lab 5 code but need to modify quaternion comparison)

        x, y, z, qx, qy, qz, qw = self.goal_pose.pose.position.x, self.goal_pose.pose.position.y, self.goal_pose.pose.position.z, self.goal_pose.pose.orientation.x, self.goal_pose.pose.orientation.y, self.goal_pose.pose.orientation.z, self.goal_pose.pose.orientation.w
        x_map, y_map, z_map, qx_map, qy_map, qz_map, qw_map = self.mapPosition.pose.pose.position.x, self.mapPosition.pose.pose.position.y, self.mapPosition.pose.pose.position.z, self.mapPosition.pose.pose.orientation.x, self.mapPosition.pose.pose.orientation.y, self.mapPosition.pose.pose.orientation.z, self.mapPosition.pose.pose.orientation.w

        angle_diff_deg = self.compute_angle_difference(np.array([qx, qy, qz, qw]), np.array([qx_map, qy_map, qz_map, qw_map]))
        dist_diff = math.dist([x, y], [x_map, y_map])

        goal_dist_tolerance_nav2 = 0.10
        goal_dist_tolerance_amcl = 0.30
        goal_angle_tolerance = 10
        
        # if (abs(x_map - x) < goal_dist_tolerance) and (abs(y_map - y) < goal_dist_tolerance): # and (angle_diff_deg < goal_angle_tolerance):
        if self.distance_remaining is not None:
            if float(self.distance_remaining) < goal_dist_tolerance_nav2 and (dist_diff < goal_dist_tolerance_amcl):

                self.get_logger().info('waypoint reached')

                if angle_diff_deg < goal_angle_tolerance:

                    self.maze_position = self.goal_position[0]
                    self.current_direction = self.goal_position[1]
                    self.goal_position = None

                    self.get_logger().info(f'current position: {self.maze_position}, current direction: {self.current_direction}')

                    return True
                
                self.get_logger().info(f'still travelling to waypoint from {self.maze_position} {self.current_direction}, {self.goal_position} not reached yet')
                self.get_logger().info(f'waiting for robot to turn, angle diff: {angle_diff_deg}')
                self.publish_position(x, y, z, qx, qy, qz, qw)
                return False

            self.get_logger().info(f'still travelling to waypoint from {self.maze_position} {self.current_direction}, {self.goal_position} not reached yet')
            # self.get_logger().info(f'x: {x_map}, y: {y_map}, z: {z_map}, qx: {qx_map}, qy: {qy_map}, qz: {qz_map}, qw: {qw_map}')
            # self.get_logger().info(f'goal x: {x}, goal y: {y}, goal z: {z}, goal qx: {qx}, goal qy: {qy}, goal qz: {qz}, goal qw: {qw}')
            self.get_logger().info(f'dist diff: {self.distance_remaining}, angle diff: {angle_diff_deg}')

            self.publish_position(x, y, z, qx, qy, qz, qw)

            return False
        
        self.get_logger().info('waiting for feedback, cannot check waypoint reached')
        self.publish_position(x, y, z, qx, qy, qz, qw)
        return False

def main():
    rclpy.init()
    navigate_maze = NavigateMaze()
    localize = True
    
    while rclpy.ok():

        try:
            
            rclpy.spin_once(navigate_maze)
            
            if localize:
                localize = navigate_maze.localize()
            
            else:
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