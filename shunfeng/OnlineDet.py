#!/usr/bin/env python

import cv2
import time
import rospy
import numpy as np
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image
from datetime import datetime
from std_srvs.srv import SetBool, SetBoolResponse  # 正确的服务类型

from ultralytics import YOLO
from plot import draw_arrow

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

def draw_grasp(img, R, T, S, K, D=np.zeros(5), color=(0,0,255)):
    grasp_points = np.float32([[0,0,0], [-S[0],0,0], [-S[0],S[1],0]])
    image_points, _ = cv2.projectPoints(grasp_points, R, T, K, D)
    image_points = np.int32(image_points).reshape(-1, 2)
    cv2.line(img, tuple(image_points[0]), tuple(image_points[1]), color, 2)  # 
    cv2.line(img, tuple(image_points[1]), tuple(image_points[2]), color, 2)  # 



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
        self.pub_img = rospy.Publisher('/perception/object_detection/display_DetImg', Image, queue_size=2)
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

        yolo_output = yolo(rgb, conf=0.45, iou=0.5, verbose=False)[0]
        if len(yolo_output) != 0:
            kpts = yolo_output.keypoints.xy.cpu().numpy()
            idx_conf = (yolo_output.keypoints.conf.cpu().numpy() > 0.8).all(axis=1)  # n,4 -> n
            kpts = kpts[idx_conf]
            if len(kpts) != 0:
                Rmt_icp_all = np.empty((len(kpts), 4, 4))
                for _idx, _kpt in enumerate(kpts):
                    for _p in _kpt.astype(int):
                        cv2.circle(rgb, _p, 3, (0,0,255),-1)
                    ret, Rm_icp, t_icp = cv2.solvePnP(self.objp, _kpt, K, np.zeros(5))
                    Rmt_icp_all[_idx, :3, :3] = cv2.Rodrigues(Rm_icp)[0]
                    Rmt_icp_all[_idx, :3, 3:] = t_icp
                    Rmt_icp_all[_idx, 3, :] = (0,0,0,1)
                    draw_arrow(rgb, Rmt_icp_all[_idx, :3, :3], Rmt_icp_all[_idx, :3, 3:], K, line_length=62)
                    for grasp in self.grasps:
                        _g = np.eye(4)
                        _g[:3,3] = grasp[0]
                        _g = Rmt_icp_all[_idx] @ _g
                        draw_grasp(rgb, _g[:3, :3], _g[:3, 3:], grasp[1], K)

        # publish result
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(rgb, 'bgr8'))


if __name__ == '__main__':

    ASSETS = '/home/kepler/catkin_ws/src/perception/assets'
    yolo = YOLO(ASSETS + '/yolov8s_ShTmp.pt')
    print('Model loaded!')


    rospy.init_node('object_dection', anonymous=True)
    object_dection = ObjectDection()
    rospy.spin()



