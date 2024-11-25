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
import time

class NavigateMaze(Node):

    def __init__(self):		

        super().__init__('navigate_maze')
		
        #Set up QoS Profiles for passing numbers over WiFi
        num_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.RELIABLE,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

    #     self.obstacle_range_robot_pose_subscriber = self.create_subscription(
	# 			Float32MultiArray,
	# 			'/obstacle_robot_info/obstacle_dist_robot_pose',
    #             self.obstacle_range_robot_pose_callback,
	# 			num_qos_profile)
    #     self.obstacle_range_robot_pose_subscriber # Prevents unused variable warning.

    #     # create velocity publisher
    #     self.velocity_publisher = self.create_publisher(
	# 			Twist, 
	# 			'/cmd_vel',
	# 			num_qos_profile)
    #     self.velocity_publisher

    # def obstacle_range_robot_pose_callback(self, msg):
    #     # self.get_logger().info('receiving robot pose')
    #     if len(msg.data) == 7:
    #         self.globalPos.x, self.globalPos.y, self.globalAng, self.globalAng_deg, self.obstacle_dected, self.min_range_angle, self.min_range = msg.data
    #         self.get_logger().info('received global pose and obstacle info (x, y, ang_rad, ang_deg, obstacle_detected, min_range_angle, min_range): "%s"' % msg.data)

    # def check_state_reached(self):

    #     diff = math.dist(self.goal_coords, (self.globalPos.x, self.globalPos.y))

    #     if diff < self.position_tolerance:

    #         self.get_logger().info(f'state {self.state} reached!')

    #         return True

    #     return False

    # def check_navigation_complete(self):

    #     all_goals = set(self.goals_reached)
        
    #     if 3 in all_goals:
    #         self.get_logger().info(f'navigation complete!')
    #         return True
        
    #     return False

    def get_state(self, waypoint_reached=False, classification_result=None):
        
        ### classification classes:
        # 0: nothing/other
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

        if waypoint_reached is True:

            # did not classify direction sign yet
            if classification_result is None:
                
                self.state = 0

            # classified direction sign
            else: 

                # goal reached
                if classification_result == 5:
                    self.state = 2

                # did not reach goal yet, follow direction sign
                else:
                    self.state = 1

        if self.state == 0:
            self.get_logger().info(f'classifying direction sign')
            self.classify_direction_sign()

        elif self.state == 1:
            self.get_logger().info(f'traveling to next waypoint')
            self.navigate_to_waypoint()

        elif self.state == 2:
            self.get_logger().info(f'goal reached!')
            self.stop()

            return True

        return False
    
    def classify_direction_sign(self):

        # get classification result
        classification_result = self.get_classification_result()

        # update state
        self.get_state(waypoint_reached=True, classification_result=classification_result)

    def check_wall_following(self):

        if self.wall_following is True:

            if self.wall_following_start_location is None:
                self.wall_following_start_location = (self.globalPos.x, self.globalPos.y)

            else:

                dx = abs(self.globalPos.x - self.wall_following_start_location[0])
                dy = abs(self.globalPos.y - self.wall_following_start_location[1])

                diff = math.dist((self.globalPos.x, self.globalPos.y), (self.wall_following_start_location))

                if (dx > self.box_thickness and dy < self.coordinate_noise) or (dx < self.coordinate_noise and dy > self.box_thickness) or (diff > self.box_thickness):
                    self.wall_following = False
                    self.wall_following_start_location = None


    def stop(self):
        self.linear_x_vel = 0.0
        self.angular_z_vel = 0.0
        self.publish_velocity()
        self.get_logger().info('stopping')
        time.sleep(10)

    def get_turn_direction_and_angle_diff(self):

        ### NEED TO FIX ANGLE DIFF

        noise = 3

        dx = self.goal_coords[0] - self.globalPos.x
        dy = self.goal_coords[1] - self.globalPos.y

        angle_between_curr_and_goal = math.degrees(math.atan2(dy, dx))
        if angle_between_curr_and_goal < 0:
            angle_between_curr_and_goal += 360
        self.get_logger().info(f'angle between curr and goal: {angle_between_curr_and_goal}') # state 1 - 0, state 2 - 90, state 3 - 180
        angle_diff_between_curr_and_goal = angle_between_curr_and_goal - self.globalAng_deg
        self.get_logger().info(f'angle diff between curr and goal: {angle_diff_between_curr_and_goal}')

        # angle_diff_between_curr_and_goal = self.goal_angle - self.globalAng_deg
        # self.get_logger().info(f'goal angle: {self.goal_angle}, curr angle: {self.globalAng_deg}')
        # self.get_logger().info(f'angle diff between curr and goal: {angle_diff_between_curr_and_goal}')

        if angle_diff_between_curr_and_goal > 360:
            angle_diff_between_curr_and_goal -= 360

        # if noise < angle_diff_between_curr_and_goal < 180:
        #     self.direction = 1 # left
        #     angle_diff = angle_diff_between_curr_and_goal - noise
        
        # elif noise < angle_diff_between_curr_and_goal < (360 - noise):
        #     self.direction = -1 # right
        #     angle_diff = (360 - noise) - angle_diff_between_curr_and_goal

        # else:
        #     self.direction = 0
        #     angle_diff = 0

        if (360 - noise) > abs(angle_diff_between_curr_and_goal) > noise:

            if angle_diff_between_curr_and_goal < 0: # want to turn right
                self.direction = -1 # right
                angle_diff = angle_diff_between_curr_and_goal

            else:
                self.direction = 1 # left
                angle_diff = angle_diff_between_curr_and_goal

        else:
            self.direction = 0
            angle_diff = 0

        self.get_logger().info(f'angle diff: {angle_diff}')

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

        noise = 0.05

        diff = math.dist(self.goal_coords, (self.globalPos.x, self.globalPos.y)) # positive if too close, negaitve if too far
        self.get_logger().info(f'distance to goal: {diff}')

        speed = round(self.Kp_dist * diff, 2)

        if speed > 0:
            speed = min(self.pos_top_speed, speed)
        
        else:
            speed = max(self.neg_top_speed, speed)

        if diff > noise or diff < -noise:
            self.linear_x_vel = speed

        else:
            self.linear_x_vel = 0.0

        self.linear_x_vel = float(self.linear_x_vel)
        self.get_logger().info(f'linear velocity: {self.linear_x_vel}')

    def navigate_to_waypoint(self):

        self.get_angular_velocity()
        self.get_linear_velocity()

        # adjust angle first
        if self.angular_z_vel > 2:
            self.linear_x_vel = 0.0

        self.publish_velocity()

    def get_spin_velocity(self):
        self.get_spin_direction()
        speed = 1
        self.vel = float(self.direction * speed)

    def publish_velocity(self):
        vel = Twist()
        vel.linear.x = self.linear_x_vel
        vel.angular.z = self.angular_z_vel
        self.velocity_publisher.publish(vel)
        self.get_logger().info('velocity: "%s"' % vel)	

    def get_user_input(self):
        return self._user_input

def main():
    rclpy.init() #init routine needed for ROS2.
    go_to_goal = GoToGoal() #Create class object to be used.
    
    while rclpy.ok():

        try:
            rclpy.spin_once(go_to_goal) # Trigger callback processing.
            navigation_complete = go_to_goal.get_state()

            if navigation_complete is True:
                break

        except KeyboardInterrupt:
            break

	#Clean up and shutdown.

    go_to_goal.destroy_node()  
    rclpy.shutdown()


if __name__ == '__main__':
	main()