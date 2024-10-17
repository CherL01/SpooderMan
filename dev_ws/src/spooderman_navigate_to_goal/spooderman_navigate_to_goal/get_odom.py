# Yi Lian
# Evan Rosenthal

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import sys

import numpy as np
import cv2
from cv_bridge import CvBridge

from std_msgs.msg import Int64

class GetObjectRange(Node):

	def __init__(self):	
		# Creates the node.
		super().__init__('object_range_detector')


		##### get range and orientation of obstacle, need to plan ####

		#Set up QoS Profiles for passing numbers over WiFi
		num_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.RELIABLE,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

		# create center coordinate publisher
		self.coordinate_publisher = self.create_publisher(
				Int64, 
				'/object_detect/coords',
				num_qos_profile)
		self.coordinate_publisher

	def send_coords(self):

		if self.center != None:
			x, y = self.center
			msg = Int64()
			msg.data = x
			self.coordinate_publisher.publish(msg)
			self.get_logger().info('center (x-coord): "%s"' % msg.data)	

	def get_image(self):
		return self._imgBGR

	def show_image(self, img):
		cv2.imshow(self._titleOriginal, img)
		# Cause a slight delay so image is displayed
		self._user_input=cv2.waitKey(50) #Use OpenCV keystroke grabber for delay.

	def get_user_input(self):
		return self._user_input

	def track_location(self, img):

		### CHERRY'S CODE ###
		# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		# yellow_lower = np.array([20, 100, 100])
		# yellow_upper = np.array([30, 255, 255])
		# mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
		# contours, _ = cv2.findContours(mask_yellow.copy(), 
		# 								cv2.RETR_TREE, 
		# 								cv2.CHAIN_APPROX_SIMPLE)
		# if len(contours) > 0:
		# 	yellow_area = max(contours, key=cv2.contourArea)
		# 	x, y, w, h = cv2.boundingRect(yellow_area)
		# 	cv2.rectangle(img,(x, y),(x+w, y+h),(0, 0, 255), 2)
		# 	self.show_image(img)
		### END OF CHERRY'S CODE ###

		light_yellow_lower = np.array([20,100,100])
		light_yellow_upper = np.array([30,255,255])

		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		mask = cv2.inRange(hsv_img, light_yellow_lower, light_yellow_upper)

		# Find contours in the mask
		contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# # Loop over contours
		# for contour in contours:
		# 	# Calculate the area and ignore small areas to reduce noise
		# 	area = cv2.contourArea(contour)
		# 	if area > 500:

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

		else:
			half = 320 // 2
			self.center = (half, half)

		# # Display the result
		# self.show_image(img)

def main():
	rclpy.init() #init routine needed for ROS2.
	object_detector = GetObjectRange() #Create class object to be used.

	while rclpy.ok():
		
		try:
			rclpy.spin_once(object_detector) # Trigger callback processing.

            # track location of object
			object_detector.track_location(object_detector.get_image()) 

            # publish x coordinate of object
			object_detector.send_coords()
			
		except KeyboardInterrupt:
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
