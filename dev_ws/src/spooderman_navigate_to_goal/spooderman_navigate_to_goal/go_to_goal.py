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

class GoToGoal(Node):

    def __init__(self):		
		# Creates the node.
        super().__init__('go_to_goal')
        self.globalPos = Point()
        self.globalAng = 0.0
        self.globalAng_deg = 0.0

        self.direction = 0
        self.pos_top_speed = 0.21 # m/s
        self.neg_top_speed = -0.21
        self.Kp_dist = 0.75
        self.Kp_angle = 0.05
        self.pos_top_angular_speed = 2.5 # rad/s
        self.neg_top_angular_speed = -2.5

        self.linear_x_vel = 0.0
        self.angular_z_vel = 0.0

        self.state = 1
        self.previous_state = None
        self.waypoint1_coords = (1.5, 0.0)
        self.waypoint2_coords = (1.5, 1.4)
        self.waypoint3_coords = (0.0, 1.4)
        self.goals = [self.waypoint1_coords, self.waypoint2_coords, self.waypoint3_coords]
        self.position_tolerance = 0.08
        self.goal_num = 0
        self.goal_coords = self.goals[self.goal_num]
        self.goals_reached = []
        self.goal_angles = [0, 90, 180]
        self.goal_angle = self.goal_angles[self.goal_num]

        self.linear_x_vel = 0.0
        self.angular_z_vel = 0.0

        self.obstacle_dected, self.min_range_angle, self.min_range = 0.0, 0.0, 0.0
        self.obstacle_angular_z_vel = 0.5
        self.obstacle_linear_x_vel = 0.05
        self.wall_following = True
        self.wall_following_start_location = None
        self.box_thickness = 0.7
        self.coordinate_noise = 0.05
        self.wall_following_distance = 0.2
        self.emergency_backup_distance = 0.1
		
        #Set up QoS Profiles for passing numbers over WiFi
        num_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.RELIABLE,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

        # self.global_position_subscriber = self.create_subscription(
		# 		Float32MultiArray,
		# 		'/robot_position/global_pose',
        #         self.global_position_callback,
		# 		num_qos_profile)
        # self.global_position_subscriber # Prevents unused variable warning.

        self.obstacle_range_robot_pose_subscriber = self.create_subscription(
				Float32MultiArray,
				'/obstacle_robot_info/obstacle_dist_robot_pose',
                self.obstacle_range_robot_pose_callback,
				num_qos_profile)
        self.obstacle_range_robot_pose_subscriber # Prevents unused variable warning.

        # create velocity publisher
        self.velocity_publisher = self.create_publisher(
				Twist, 
				'/cmd_vel',
				num_qos_profile)
        self.velocity_publisher

    # def global_position_callback(self, msg):
    #     # self.get_logger().info('receiving robot pose')
    #     self.globalPos.x, self.globalPos.y, self.globalAng, self.globalAng_deg = msg.data
    #     # self.get_logger().info('received global pose is x:{}, y:{}, a:{}, a_deg'.format(self.globalPos.x,self.globalPos.y,self.globalAng, self.globalAng_deg))

    def obstacle_range_robot_pose_callback(self, msg):
        # self.get_logger().info('receiving robot pose')
        if len(msg.data) == 7:
            self.globalPos.x, self.globalPos.y, self.globalAng, self.globalAng_deg, self.obstacle_dected, self.min_range_angle, self.min_range = msg.data
            self.get_logger().info('received global pose and obstacle info (x, y, ang_rad, ang_deg, obstacle_detected, min_range_angle, min_range): "%s"' % msg.data)

    def check_state_reached(self):

        diff = math.dist(self.goal_coords, (self.globalPos.x, self.globalPos.y))

        if diff < self.position_tolerance:

            self.get_logger().info(f'state {self.state} reached!')

            return True

        return False

    def check_navigation_complete(self):

        all_goals = set(self.goals_reached)
        
        if 3 in all_goals:
            self.get_logger().info(f'navigation complete!')
            return True
        
        return False

    def get_state(self):
        
        # take in obstacle detection data
        # if no obstacle detected, set state to 1, 2, 3 (for each waypoint)
        # if obstacle detected, set statetoler to 0 (if avoid obstacle)
        # if navigation completed or emergency, set state to 4 (if stop)
        # state -1: stop
        # state 0: avoid obstacle 
        # state 1: navigate to waypoint 1
        # state 2: navigate to waypoint 2
        # state 3: navigate to waypoint 3
        # state 4: stop

        # check if obstacle is detected
        if self.obstacle_dected == 1.0: # obstacle detected is true

            self.check_wall_following()
            self.get_logger().info(f'wall following: {self.wall_following}')

            if self.wall_following is True:

                if self.previous_state is None:
                    self.previous_state = self.state

                self.state = 0

            elif self.previous_state is not None:
                self.state = self.previous_state

        elif self.previous_state is not None:
            self.state = self.previous_state

        if self.state == 0:
            self.avoid_obstacle()

        if self.check_state_reached():
            self.goals_reached.append(self.state)
            self.state = 4

        elif self.state == 1:

            self.previous_state = None
            self.wall_following = True

            self.goal_coords = self.waypoint1_coords
            self.navigate_to_waypoint()

        elif self.state == 2:

            self.previous_state = None
            self.wall_following = True

            self.goal_coords = self.waypoint2_coords
            self.navigate_to_waypoint()

        elif self.state == 3:

            self.previous_state = None
            self.wall_following = True

            self.goal_coords = self.waypoint3_coords
            self.navigate_to_waypoint()

        if self.state == 4:

            self.previous_state = None
            self.wall_following = True

            self.stop()

            if self.check_navigation_complete():
                # self.stop()
                return True

            # self.stop()

            self.goal_num += 1
            self.goal_coords = self.goals[self.goal_num]
            self.goal_angle = self.goal_angles[self.goal_num]
            self.state = self.goal_num + 1

            self.get_logger().info(f'now in state: {self.state}')


        self.get_logger().info(f'state: {self.state}, curr position: {self.globalPos.x, self.globalPos.y}, goal position: {self.goal_coords}')

        return False
    
    def avoid_obstacle(self):

        # wall following?

        # if self.min_range_angle < noise or self.min_range_angle > (360-noise):
        #     self.min_range_angle = noise

        # if noise < self.min_range_angle < 180: # obstacle closer to left side of robot, want to turn right
        #     goal_angle = 90
        #     direction = 1

        # else: # obstacle closer to the right side of robot, want to turn left
        #     goal_angle = 270
        #     direction = -1

        linear_noise = 0.02
        angular_noise = 3

        if self.min_range <= self.emergency_backup_distance:

            self.linear_x_vel = -self.obstacle_linear_x_vel

        else:

            if self.min_range >= (self.wall_following_distance - linear_noise) or (350 > self.min_range_angle > 10):

                goal_angle = 90

                if 100 > self.min_range_angle > 90:
                    goal_angle = self.min_range_angle - angular_noise

                angle_diff = goal_angle - self.min_range_angle

                self.get_logger().info(f'goal angle, angle diff while wall following: {goal_angle}, {angle_diff}')

                if abs(angle_diff) < angular_noise:
                    angle_diff = 0

                if angle_diff >= 0: 
                    direction = -1

                elif angle_diff < 0:
                    direction = 1

                self.get_logger().info(f'angle diff while wall following (after noise reduction): {angle_diff}')

                speed = round(self.Kp_angle * abs(angle_diff), 1)

                if speed > 0:
                    speed = min(self.obstacle_angular_z_vel, speed)
                
                else:
                    speed = max(-self.obstacle_angular_z_vel, speed)

                self.angular_z_vel = float(direction * speed)
                self.get_logger().info(f'angular velocity: {self.angular_z_vel}')

            diff = - self.wall_following_distance + self.min_range # positive if too far, negaitve if too close
            self.get_logger().info(f'distance to wall: {diff}')

            speed = round(self.Kp_dist * diff, 2)

            if speed > 0:
                speed = min(self.obstacle_linear_x_vel, speed)
            
            else:
                speed = max(-self.obstacle_linear_x_vel, speed)

            if diff > linear_noise or diff < -linear_noise:
                self.linear_x_vel = speed

            else:
                self.linear_x_vel = self.obstacle_linear_x_vel

            self.linear_x_vel = float(self.linear_x_vel)
            self.get_logger().info(f'linear velocity: {self.linear_x_vel}')

            # self.linear_x_vel = self.obstacle_linear_x_vel

        self.publish_velocity()

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