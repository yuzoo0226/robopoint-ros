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


class RoboPointVQAUtils:
    def __init__(self):
        rospy.init_node("test_get_place_pose")
        self.bridge = CvBridge()

        rospy.wait_for_service("/robopoint/get_place_pose")
        self.get_place_pose_service = rospy.ServiceProxy("/robopoint/get_place_pose", GetPlacePose)

    def call_get_place_pose(self, image_path: str, prompt: str, fx: float = 1.0, fy: float = 1.0):
        """Call the get_place_pose service with the given image path.

        Args:
            image_path (str): The path to the image file.
        """
        req = GetPlacePoseRequest()
        if not os.path.exists(image_path):
            rospy.logerr(f"Image path does not exist: {image_path}")
            return
        cv_image = cv2.imread(image_path, 1)
        cv_image = cv2.resize(cv_image, dsize=None, fx=fx, fy=fy)
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

        req.image = ros_image
        req.prompt = prompt
        req.temperature = 0.2  # Set a default temperature value

        try:
            resp = self.get_place_pose_service(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def spin(self):
        rospy.spin()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    node = RoboPointVQAUtils()
    # prompt = "Identify several spots within the vacant space near the orange fanta on the cabinet. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
    # prompt = "Find the vacant space to the left of the spam can. Your answer should be a tuple (min_x, max_x, min_y, max_y) indicating the bounding box of the target region. min_x, max_x, min_y, max_y are normalized image coordinates between 0 and 1."
    # node.call_get_place_pose(image_path="/home/hma/usr/ros_ws/src/5_skills/robopoint-ros/io/test_images/test_scene1.jpg", prompt=prompt)

    prompt = "Identify several spots within the vacant space on the white dishwasher. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
    node.call_get_place_pose(image_path="/home/hma/usr/ros_ws/src/5_skills/robopoint-ros/io/test_images/test_scene3.jpg", prompt=prompt, fx=0.5, fy=0.5)
    node.spin()
