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
        self.waypoint1_coords = (1.5, 0.0)
        self.waypoint2_coords = (1.5, 1.4)
        self.waypoint3_coords = (0.0, 1.4)
        self.goals = [self.waypoint1_coords, self.waypoint2_coords, self.waypoint3_coords]
        self.position_tolerance = 0.1
        self.goal_num = 0
        self.goal_coords = self.goals[self.goal_num]
        self.goals_reached = []
        self.goal_angles = [0, 90, 180]
        self.goal_angle = self.goal_angles[self.goal_num]

        self.linear_x_vel = 0.0
        self.angular_z_vel = 0.0
		
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

        # create velocity publisher
        self.velocity_publisher = self.create_publisher(
				Twist, 
				'/cmd_vel',
				num_qos_profile)
        self.velocity_publisher

    def global_position_callback(self, msg):
        # self.get_logger().info('receiving robot pose')
        self.globalPos.x, self.globalPos.y, self.globalAng, self.globalAng_deg = msg.data
        # self.get_logger().info('received global pose is x:{}, y:{}, a:{}, a_deg'.format(self.globalPos.x,self.globalPos.y,self.globalAng, self.globalAng_deg))

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

        # self.state = 1

        if self.state == 0:
            self.avoid_obstacle()

        if self.check_state_reached():
            self.goals_reached.append(self.state)
            self.state = 4

        elif self.state == 1:
            self.goal_coords = self.waypoint1_coords
            self.navigate_to_waypoint()

        elif self.state == 2:
            self.goal_coords = self.waypoint2_coords
            self.navigate_to_waypoint()

        elif self.state == 3:
            self.goal_coords = self.waypoint3_coords
            self.navigate_to_waypoint()

        elif self.state == 4:

            if self.check_navigation_complete():
                self.stop()
                return True

            self.stop()

            self.goal_num += 1
            self.goal_coords = self.goals[self.goal_num]
            self.goal_angle = self.goal_angles[self.goal_num]
            self.state = self.goal_num + 1

            self.get_logger().info(f'now in state: {self.state}')


        self.get_logger().info(f'state: {self.state}, curr position: {self.globalPos.x, self.globalPos.y}, goal position: {self.goal_coords}')

        return False
    
    def avoid_obstacle(self):
        pass

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