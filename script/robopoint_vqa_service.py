#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import ast
import math
import json
import rospy
import torch
import roslib
import argparse
import shortuuid
from tqdm import tqdm
from PIL import Image as PILImage

from robopoint.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from robopoint.conversation import conv_templates, SeparatorStyle
from robopoint.model.builder import load_pretrained_model
from robopoint.utils import disable_torch_init
from robopoint.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from robopoint_ros.srv import GetPlacePose, GetPlacePoseResponse, GetPlacePoseRequest


class RoboPointVQAService:
    def __init__(self):

        # v1
        # self.p_model_path = rospy.get_param("~model_path", "wentao-yuan/robopoint-v1-vicuna-v1.5-13b")
        # self.p_model_base = rospy.get_param("~model_base", None)

        # v2
        self.p_model_path = rospy.get_param("~model_path", "wentao-yuan/robopoint-v1-vicuna-v1.5-7b-lora")
        self.p_model_base = rospy.get_param("~model_base", "lmsys/vicuna-7b-v1.5")

        self.p_conv_mode = rospy.get_param("~conv_mode", "llava_v1")
        self.p_top_p = rospy.get_param("~top_p", 5)
        self.p_num_beams = rospy.get_param("~num_beams", 1)
        self.p_load_8bit = rospy.get_param("~load_8bit", False)
        self.p_load_4bit = rospy.get_param("~load_4bit", True)

        disable_torch_init()
        self.model_path = os.path.expanduser(self.p_model_path)
        self.model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.model_path, self.p_model_base, self.model_name, self.p_load_8bit, self.p_load_4bit)

        self.bridge = CvBridge()

        self.robopoint_package_dir = roslib.packages.get_pkg_dir("robopoint_ros")
        self.get_pose_service = rospy.Service("/robopoint/get_place_pose", GetPlacePose, self.get_place_pose)
        rospy.loginfo(f"RoboPoint VQA Service initialized with model: {self.p_model_path}, conv_mode: {self.p_conv_mode}, top_p: {self.p_top_p}, num_beams: {self.p_num_beams}")

    @staticmethod
    def split_list(lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)  # integer division
        return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

    @staticmethod
    def parse_output_string(output_str):
        """ Parse a string representation of a list of 2D tuples.

        Parameters:
        - output_str (str): e.g. "[(0.5, 0.4), (0.4, 0.3)]"

        Returns:
        - list of tuples: [(0.5, 0.4), (0.4, 0.3)]
        """
        try:
            result = ast.literal_eval(output_str)
            if isinstance(result, list) and all(isinstance(p, tuple) and len(p) == 2 for p in result):
                return result
            else:
                raise ValueError("Parsed result is not a list of 2D tuples.")
        except Exception as e:
            print(f"[ERROR] Failed to parse output string: {e}")
            return []

    def imgmsg_to_pil(self, img_msg: Image, desired_encoding='bgr8'):
        """Convert a ROS Image message to a PIL Image."""
        cv_bgr = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding=desired_encoding)
        cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(cv_rgb)

    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        return chunks[k]

    def get_place_pose(self, req: GetPlacePoseRequest) -> GetPlacePoseResponse:
        """Evaluate the model with the given request.

        Args:
            req (GetPlacePoseRequest): The request containing input data.
        Returns:
            GetPlacePoseResponse: The response containing the model's output.
        """

        qs = req.prompt

        if DEFAULT_IMAGE_TOKEN not in qs:
            cur_prompt = qs
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        else:
            cur_prompt = qs.split('\n', 1)[1]

        rospy.loginfo(f"cur_prompt: {cur_prompt}")
        rospy.loginfo(f"qs: {qs}")

        conv = conv_templates[self.p_conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        pil_image = self.imgmsg_to_pil(req.image)
        cv_image = self.bridge.imgmsg_to_cv2(req.image, desired_encoding="bgr8")

        image_tensor = process_images([pil_image], self.image_processor, self.model.config)[0]

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[pil_image.size],
                do_sample=True if req.temperature > 0 else False,
                temperature=req.temperature,
                top_p=self.p_top_p,
                num_beams=self.p_num_beams,
                max_new_tokens=1024,
                use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        outputs_list = self.parse_output_string(outputs)
        rospy.loginfo(f"outputs: {type(outputs_list)}, {outputs_list}")

        height, width = cv_image.shape[:2]
        for output in outputs_list:
            cv2.circle(cv_image, (int(output[0]*width), int(output[1]*height)), 5, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(self.robopoint_package_dir, "io/test_images/temp_result.png"), cv_image)

        response = GetPlacePoseResponse()
        # response.pointarray = outputs

        return response


if __name__ == "__main__":
    rospy.init_node("tracking_node")
    node = RoboPointVQAService()
    rospy.spin()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="robopoint-v1-vicuna-v1.5-13b")
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-folder", type=str, default="")
    # parser.add_argument("--question-file", type=str, default="question.jsonl")
    # parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    # parser.add_argument("--conv-mode", type=str, default="llava_v1")
    # parser.add_argument("--num-chunks", type=int, default=1)
    # parser.add_argument("--chunk-idx", type=int, default=0)
    # parser.add_argument("--temperature", type=float, default=0.2)
    # parser.add_argument("--top_p", type=float, default=None)
    # parser.add_argument("--num_beams", type=int, default=1)
    # args = parser.parse_args()

    # eval_model(args)
