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
        self.Kp_dist = 3
        self.Kp_angle = 0.08
        self.pos_top_angular_speed = 2.8 # rad/s
        self.neg_top_angular_speed = -2.8

        self.linear_x_vel = 0.0
        self.angular_z_vel = 0.0

        self.state = 4
        self.waypoint1_coords = (1.5, 0.0)
        self.waypoint2_coords = (1.5, 1.4)
        self.waypoint3_coords = (0.0, 1.4)
        self.positon_tolerance = 0.1
		
        #Set up QoS Profiles for passing numbers over WiFi
        num_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.RELIABLE,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

		# Declare that the velocity_generator node is subcribing to the /object_detect/coords topic.
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
        self.globalPos, self.globalAng, self.globalAng_deg = msg.data
        self.get_logger().info('received global pose is x:{}, y:{}, a:{}'.format(self.globalPos.x,self.globalPos.y,self.globalAng))

    def get_state(self):
        
        # take in obstacle detection data
        # if no obstacle detected, set state to 1, 2, 3 (for each waypoint)
        # if obstacle detected, set state to 0 (if avoid obstacle)
        # if navigation completed or emergency, set state to 4 (if stop)
        # state -1: stop
        # state 0: avoid obstacle 
        # state 1: navigate to waypoint 1
        # state 2: navigate to waypoint 2
        # state 3: navigate to waypoint 3
        # state 4: stop

        self.state = 1

        if self.state == 0:
            self.avoid_obstacle()

        elif self.state == 1:
            self.navigate_to_waypoint(1)

        elif self.state == 2:
            self.navigate_to_waypoint(2)

        elif self.state == 3:
            self.navigate_to_waypoint(3)

        elif self.state == 4:
            self.stop()
    
    def avoid_obstacle(self):
        pass

    def stop(self):
        vel = Twist()
        vel.linear.x = 0.0
        vel.angular.z = 0.0
        self.velocity_publisher.publish(vel)
        self.get_logger().info('stopping')

    def get_turn_direction_and_angle_diff(self):

        noise = 5

        if noise < self.globalAng_deg < 180:
            self.direction = 1 # left
            angle_diff = self.globalAng_deg - noise
        
        elif noise < self.globalAng_deg < (360 - noise):
            self.direction = -1 # right
            angle_diff = (360 - noise) - self.globalAng_deg

        else:
            self.direction = 0
            angle_diff = 0

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

        noise = 0.01

        diff = self.distance - self.target_distance # positive if too close, negaitve if too far

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

    def navigate_to_waypoint(self, waypoint):

        if waypoint == 1:
            x, y = self.waypoint1_coords

        elif waypoint == 2:
            x, y = self.waypoint2_coords

        else: 
            x, y = self.waypoint3_coords

        self.get_angular_velocity()
        self.get_linear_velocity()
        


    def get_spin_velocity(self):
        self.get_spin_direction()
        speed = 1
        self.vel = float(self.direction * speed)

    def publish_spin_velocity(self):
        if self.x_coord != None:
            self.get_spin_velocity()
            vel = Twist()
            vel.angular.z = self.vel
            self.velocity_publisher.publish(vel)
            self.get_logger().info('angular velocity: "%s"' % vel.angular.z)	

    def get_user_input(self):
        return self._user_input

def main():
    rclpy.init() #init routine needed for ROS2.
    velocity_generator = VelocityGenerator() #Create class object to be used.
    
    while rclpy.ok():

        try:
            rclpy.spin_once(velocity_generator) # Trigger callback processing.
            velocity_generator.publish_spin_velocity()

        except KeyboardInterrupt:
            break

	#Clean up and shutdown.

    velocity_generator.destroy_node()  
    rclpy.shutdown()


if __name__ == '__main__':
	main()