import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import sys

import numpy as np
import cv2
from cv_bridge import CvBridge

from std_msgs.msg import Int64, Float32, Float32MultiArray

import math

class ObjectDetector(Node):

    def __init__(self):		
		# Creates the node.
        super().__init__('object_detector')

		# Set Parameters
        self.declare_parameter('show_image_bool', False)
        self.declare_parameter('window_name', "Raw Image")

		#Determine Window Showing Based on Input
        self._display_image = bool(self.get_parameter('show_image_bool').value)

		# Declare some variables
        self._titleOriginal = self.get_parameter('window_name').value # Image Window Title	
		
		#Only create image frames if we are not running headless (_display_image sets this)
        if(self._display_image):
		# Set Up Image Viewing
            cv2.namedWindow(self._titleOriginal, cv2.WINDOW_AUTOSIZE ) # Viewing Window
            cv2.moveWindow(self._titleOriginal, 50, 50) # Viewing Window Original Location
		
		#Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(
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

		#Declare that the minimal_video_subscriber node is subcribing to the /camera/image/compressed topic.
        self._video_subscriber = self.create_subscription(
				CompressedImage,
				# '/camera/image/compressed',source ~/PaulB/[LabX]/install/setup.bash
				'/image_raw/compressed',
				self._image_callback,
				image_qos_profile)
        self._video_subscriber # Prevents unused variable warning.

		# create center coordinate publisher
        self.coordinate_publisher = self.create_publisher(
				Float32MultiArray, 
				'/object_detect/x_coords',
				num_qos_profile)
        self.coordinate_publisher
		
        # create center coordinate publisherAttributeError: 'numpy.ndarray' object has no attribute 'shapeFloat32'
        self.angle_publisher = self.create_publisher(
				Float32MultiArray, 
				'/object_detect/angles',
				num_qos_profile)
        self.angle_publisher

		# additional variables added
        self.center = None
        self.radius = None
        self.x_coords = None
        self.height = None
        self.width = None
        self.channels = None
        self.fov = 62.2 # deg

    def _image_callback(self, CompressedImage):	
		# The "CompressedImage" is transformed to a color image in BGR space and is store in "_imgBGR"
        self._imgBGR = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")
        self.height, self.width, self.channels = self._imgBGR.shape
        # self.get_logger().info(f'image size (height, width, channels): {self.height}, {self.width}, {self.channels}')	
		
        if(self._display_image):
			# Display the image in a window
            self.show_image(self._imgBGR)

    def send_x_coord(self):

        if self.center != None:
            msg = Float32MultiArray()
            msg.data = self.x_coords
            self.coordinate_publisher.publish(msg)
            self.get_logger().info(f'x coords: {self.x_coords}')	
			
    def send_angle(self):
	
        if self.center != None:
            angles = [self.get_angle(x) for x in self.x_coords]
            msg = Float32MultiArray()
            msg.data = angles
            self.angle_publisher.publish(msg)
            self.get_logger().info(f'angles (deg): {angles}')	
			
    def get_angle(self, x):
            
        half = self.width / 2
        angle = (half - x) * self.fov / self.width # 62.2 deg / 320 pixels
        if angle < 0:
            angle += 360
			
        return angle

    def get_image(self):
        return self._imgBGR

    def show_image(self, img):
        cv2.imshow(self._titleOriginal, img)
		# Cause a slight delay so image is displayed
        self._user_input=cv2.waitKey(50) #Use OpenCV keystroke grabber for delay.

    def get_user_input(self):
        return self._user_input

    def track_location(self, img):

        light_yellow_lower = np.array([20,100,100])
        light_yellow_upper = np.array([30,255,255])

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_img, light_yellow_lower, light_yellow_upper)

		# Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)

			# Get the center and radius of the enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

			# Draw the circle and centroid on the img
            cv2.circle(img, center, radius, (0, 255, 0), 2)
            cv2.circle(img, center, 5, (0, 0, 255), -1)
            self.center = center
            self.radius = radius

            x, y = self.center
            self.x_coords = [float(x - self.radius), float(x), float(x + self.radius)]

        else:
            half = self.width / 2
            self.center = (half, half)
            self.radius = 0

            self.center = math.inf, math.inf
            self.x_coords = [math.inf, math.inf, math.inf]

		# Display the result
        self.show_image(img)

def main():
    rclpy.init() #init routine needed for ROS2.
    object_detector = ObjectDetector() #Create class object to be used.

    while rclpy.ok():
		
        try:
            rclpy.spin_once(object_detector) # Trigger callback processing.

            # track location of object
            object_detector.track_location(object_detector.get_image()) 

            # publish x coordinate of object
            object_detector.send_x_coord()

            # publish angle of object
            object_detector.send_angle()
			
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            break

		# if(object_detector._display_image):
			# object_detector.show_image(object_detector.get_image())	
		# if object_detector.get_user_input() == ord('q'):
		# 	cv2.destroyAllWindows()
		# 	break

	#Clean up and shutdown.
    object_detector.destroy_node()  
    rclpy.shutdown()


if __name__ == '__main__':
	main()