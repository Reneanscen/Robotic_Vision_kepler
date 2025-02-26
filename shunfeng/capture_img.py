#!/usr/bin/env python

import cv2
import time
import rospy
import numpy as np
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image
from datetime import datetime
from std_srvs.srv import SetBool, SetBoolResponse  # 正确的服务类型


from kpl_msgs.msg import ArmPoseArray, ArmPose

import sys
from pathlib import Path
CODE_DIR = str(Path(__file__).parent) + '/'
IMG_HOME = Path('/home/kepler/catkin_ws/src/perception/assets/imgs')
sys.path.append(CODE_DIR + 'devtools')
from camera_feynman_ros1 import ImageSubscriber
from general import logging, log, generate_directory
from cvkit import calculate_histogram_similarity, cal_maxpooling_similarity
# from . import SaveData

#class ManageDataSave():
#    def __init__(self):
#        self.on = True  # save data switch is default on
#        rospy.Subscriber("/perception/save_data", Bool, self.manage_callback)
#    def manage_callback(self, b_save):
#        self.on = data.b_save
#        print('save data is on: %s' % self.on)

class ManageDataSaveService():
    def __init__(self):
        self.on = True
        # ROS节点初始化
        # rospy.init_node('person_server')
        # 创建一个名为/show_person的server，注册    回调函数manage_callback
        s = rospy.Service("/perception/save_data", SetBool, self.manage_callback)
    def manage_callback(self, data):
        self.on = data.data
        print('save data is on: %s' % self.on)
        return SetBoolResponse(data.data, "Success")  # 确保返回一个响应对象


class ObjectDection(ImageSubscriber, ManageDataSaveService):
    def __init__(self):
        super().__init__()
        ManageDataSaveService.__init__(self)
        self.timer = rospy.Timer(rospy.Duration(0.5), self.perception_callback)
        generate_directory(str(IMG_HOME / 'rgb'))
        generate_directory(str(IMG_HOME / 'depth'))
        self.pub_img = rospy.Publisher('/perception/object_detection/display_img', Image, queue_size=2)
        self.pre_img = None
        self.num_saved = 0

        print('%s initiated...' % self.__class__.__name__)

    # @log
    def perception_callback(self, event):
        try:
            K = np.array(self.K).reshape(3,3)
            rgb = self.rgb.copy()
            depth = self.depth.copy()
        except AttributeError as e:
            print(e)
            return

        if abs(self.depth_stamp - self.rgb_stamp) > 0.2:
            print('Time interval between rgb and depth is too large: %.3fs' % abs(self.depth_stamp - self.rgb_stamp))

        if self.num_saved > 1000:
            return

        current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")
        if self.pre_img is None:
            self.pre_img = rgb
            cv2.imwrite(str(IMG_HOME / 'rgb/%s.jpg') % current_time, rgb)
            cv2.imwrite(str(IMG_HOME / 'depth/%s.png') % current_time, depth)
            print('Img saved...')
            self.num_saved += 1
            return

        similarity = cal_maxpooling_similarity(self.pre_img, rgb)
        print(similarity)
        
        # Mode 1: Collect data:
        # rosservice call /perception/save_data "data: true"
        # if self.on:

        # Mode 2: Collect data:
        if similarity > 0.12:
            cv2.imwrite(str(IMG_HOME / 'rgb/%s.jpg') % current_time, rgb)
            cv2.imwrite(str(IMG_HOME / 'depth/%s.png') % current_time, depth)
            print('Img saved...')
            self.num_saved += 1
            self.pre_img = rgb
            self.on = False
        # publish result
        # self.pub_img.publish(self.bridge.cv2_to_imgmsg(rgb, 'bgr8'))


if __name__ == '__main__':
    rospy.init_node('object_dection', anonymous=True)
    object_dection = ObjectDection()
    rospy.spin()



