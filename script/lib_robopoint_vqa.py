#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import glob
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from robopoint_ros.srv import GetPlacePose, GetPlacePoseResponse, GetPlacePoseRequest


class GetPlacePoseClient:
    def __init__(self):
        rospy.init_node("test_get_place_pose")
        self.bridge = CvBridge()

        rospy.wait_for_service("/robopoint/get_place_pose")
        self.get_place_pose_service = rospy.ServiceProxy("/robopoint/get_place_pose", GetPlacePose)

    def call_get_place_pose(self, image_path: str, prompt: str):
        """Call the get_place_pose service with the given image path.

        Args:
            image_path (str): The path to the image file.
        """
        req = GetPlacePoseRequest()
        if not os.path.exists(image_path):
            rospy.logerr(f"Image path does not exist: {image_path}")
            return
        cv_image = cv2.imread(image_path, 1)
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

        req.image = ros_image
        req.prompt = prompt
        req.temperature = 0.1  # Set a default temperature value

        try:
            resp = self.get_place_pose_service(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def spin(self):
        rospy.spin()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    node = GetPlacePoseClient()
    prompt = "Identify several spots within the vacant space near the ball on the cabinet. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
    node.call_get_place_pose(image_path="/home/yuga/usr/tamhome_ws/src/RoboPoint-ros/io/test_images/test_scene1.png", prompt=prompt)
    node.spin()
