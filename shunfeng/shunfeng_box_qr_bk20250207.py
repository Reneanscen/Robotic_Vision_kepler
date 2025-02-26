#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_msgs.msg import Int32MultiArray
from ultralytics import YOLO
from collections import deque
import copy
import time
import math
from kpl_msgs.msg import ArmPoseArray, ArmPose

import sys
from pathlib import Path
CODE_DIR = str(Path(__file__).parent) + '/'
sys.path.append(CODE_DIR + './devtools')
from QR_Detect import QRDetect
from camera_feynman_ros1 import ImageSubscriber
from general import log, logging
from plot import draw_arrow
from module import Grasp
logger = logging.getLogger('Perception')


def draw_grasp(img, R, T, S, K, D=np.zeros(5), color=(0,0,255)):
    grasp_points = np.float32([[0,0,0], [-S[0],0,0], [-S[0],S[1],0]])
    image_points, _ = cv2.projectPoints(grasp_points, R, T, K, D)
    image_points = np.int32(image_points).reshape(-1, 2)
    cv2.line(img, tuple(image_points[0]), tuple(image_points[1]), color, 2)  # 
    cv2.line(img, tuple(image_points[1]), tuple(image_points[2]), color, 2)  # 


# 将图片转为HSV格式，使用颜色阈值对纯色标签进行检测
# 输入图像
# 返回带矩形框的图像，mask图，标签中心点(形状n,2, )
def detect_red_table_rectangles_filtered(image):
    # 将图像转换为HSV格式
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cX = 0
    cY = 0
    area = 0


    # 定义黄色的HSV范围
    # lower_threshold = np.array([30, 90, 80])
    # upper_threshold = np.array([60, 255, 255])
    # lower_threshold = np.array([30, 110, 85])
    # upper_threshold = np.array([50, 255, 255])
    # # 创建黄色的掩码
    # mask = cv2.inRange(hsv, lower_threshold, upper_threshold)


    # 定义红色的HSV范围
    # 红色在HSV空间中跨越0度到10度以及170度到180度
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # lower_red1 = np.array([0, 199, 73])
    # upper_red1 = np.array([7, 255, 255])
    # lower_red2 = np.array([111, 199, 73])
    # upper_red2 = np.array([180, 255, 255])
    # 创建红色的掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)



    # 对图像进行二值化处理
    _, thresh = cv2.threshold(mask, 127, 255, 0)

    # 显示
    # cv2.imshow("er", thresh)


    # 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建存放所有黄色标签中心的数组
    label_center = np.zeros((len(contours), 2), dtype=int)  # 指定数据类型为int

    label_id = 0
    # 遍历所有轮廓
    for contour in contours:
        # 计算轮廓的中心点
        M = cv2.moments(contour)
        if M["m00"]:
            cX += int(M["m10"] / M["m00"])
            cY += int(M["m01"] / M["m00"])
            cY /= 2
            cX /= 2
        # 计算轮廓面积
        area = cv2.contourArea(contour)

        if area <= 10:
            continue

        # 计算轮廓的近似多边形
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # wt: 2024.10.18,求中心点
        approx_squeeze = np.squeeze(approx, axis=1)
        approx_center = np.mean(approx_squeeze, axis=0)
        label_center[label_id] = approx_center
        label_id = label_id + 1

        # print(approx_center)

        # 如果近似多边形有4个顶点，我们认为它是一个矩形
        # 没用到
        if len(approx) == 4:
            # 画出轮廓
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)


    # print("label_center: ", label_center)
    # 过滤全零行
    # 找出非全零行
    non_zero_rows = label_center.any(axis=1)
    # 根据布尔索引选择非全零行
    label_center_filtered = label_center[non_zero_rows]
    # print("label_center_filtered: ", label_center_filtered)


	#返回黄色矩形的位置和面积方便判断图像是否能够清晰
    return image, mask, label_center_filtered  

# 将图片转为HSV格式，使用颜色阈值对纯色标签进行检测
# 输入图像
# 返回带矩形框的图像，mask图，标签中心点(形状n,2, )
def detect_red_shelf_rectangles_filtered(image):
    # 将图像转换为HSV格式
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cX = 0
    cY = 0
    area = 0


    # 定义黄色的HSV范围
    # lower_threshold = np.array([30, 90, 80])
    # upper_threshold = np.array([60, 255, 255])
    # lower_threshold = np.array([30, 110, 85])
    # upper_threshold = np.array([50, 255, 255])
    # # 创建黄色的掩码
    # mask = cv2.inRange(hsv, lower_threshold, upper_threshold)


    # 定义红色的HSV范围
    # 红色在HSV空间中跨越0度到10度以及170度到180度
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # lower_red1 = np.array([0, 199, 73])
    # upper_red1 = np.array([5, 255, 255])
    # lower_red2 = np.array([111, 199, 73])
    # upper_red2 = np.array([180, 255, 255])
    # 创建红色的掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)



    # 对图像进行二值化处理
    _, thresh = cv2.threshold(mask, 127, 255, 0)

    # 显示
    # cv2.imshow("er", thresh)


    # 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建存放所有黄色标签中心的数组
    label_center = np.zeros((len(contours), 2), dtype=int)  # 指定数据类型为int

    label_id = 0
    # 遍历所有轮廓
    for contour in contours:
        # 计算轮廓的中心点
        M = cv2.moments(contour)
        if M["m00"]:
            cX += int(M["m10"] / M["m00"])
            cY += int(M["m01"] / M["m00"])
            cY /= 2
            cX /= 2
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        # print("all area: ", area)
        if area <= 10:
            continue

        # 计算轮廓的近似多边形
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # wt: 2024.10.18,求中心点
        approx_squeeze = np.squeeze(approx, axis=1)
        approx_center = np.mean(approx_squeeze, axis=0)
        label_center[label_id] = approx_center
        label_id = label_id + 1

        # print(approx_center)

        # 如果近似多边形有4个顶点，我们认为它是一个矩形
        # 没用到
        if len(approx) == 4:
            # 画出轮廓
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)


    # print("label_center: ", label_center)
    # 过滤全零行
    # 找出非全零行
    non_zero_rows = label_center.any(axis=1)
    # 根据布尔索引选择非全零行
    label_center_filtered = label_center[non_zero_rows]
    # print("label_center_filtered: ", label_center_filtered)


	#返回黄色矩形的位置和面积方便判断图像是否能够清晰
    return image, mask, label_center_filtered  


# 输入图像
# 返回带矩形框的图像，mask图，标签中心点(形状n,2, )
def detect_yellow_rectangles(image):
    # 将图像转换为HSV格式
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cX = 0
    cY = 0
    area = 0
    # 定义黄色的HSV范围
    # lower_yellow = np.array([20, 100, 100])
    # upper_yellow = np.array([30, 255, 255])
    lower_yellow = np.array([30, 110, 85])
    upper_yellow = np.array([50, 255, 255])

    # 创建黄色的掩码
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 对图像进行二值化处理
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    # cv2.imshow("er", thresh)
    # 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建存放所有黄色标签中心的数组
    label_center = np.empty((len(contours), 2), dtype=int)  # 指定数据类型为int
    label_id = 0
    # 遍历所有轮廓
    for contour in contours:
        # 计算轮廓的中心点
        M = cv2.moments(contour)
        if M["m00"]:
            cX += int(M["m10"] / M["m00"])
            cY += int(M["m01"] / M["m00"])
            cY /= 2
            cX /= 2
        # 计算轮廓面积
        area += cv2.contourArea(contour)
        # 计算轮廓的近似多边形

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # wt: 2024.10.18,求中心点
        approx_squeeze = np.squeeze(approx, axis=1)
        approx_center = np.mean(approx_squeeze, axis=0)
        label_center[label_id] = approx_center
        label_id = label_id + 1
        # print(approx_center)

        # 如果近似多边形有4个顶点，我们认为它是一个矩形
        if len(approx) == 4:
            # 画出轮廓
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
	#返回黄色矩形的位置和面积方便判断图像是否能够清晰
    # return image, cX, cY, area
    return image, mask, label_center



# 输入：一个二维平面点C(2,)，另外一组二维平面点(n, 2) 
# 输出：包围点A形成矩形的四个角点，从右上角开始，逆时针排序
def InferCorner(center_point, points):

    # 中心点的xy
    Center_x = center_point[0]
    Center_y = center_point[1]

    # 假定以中心点为原点，建立右手坐标系
    # 计算所有黄色标签在坐标系的坐标值
    Quadrant_info = np.zeros((points.shape[0], 3))
    Quadrant_info[:, 0] = points[:, 0] - Center_x
    Quadrant_info[:, 1] = points[:, 1] - Center_y

    # 计算所有像素点距离原点的距离
    xy_columns = Quadrant_info[:, :2]
    distance2origin = np.sum(np.square(xy_columns), axis=1)

    # 根据标签点在象限的分布以及与原点的距离信息，求四个象限下距离原点最近的点
    Quadrant_idx = np.zeros(4)
    Min_distance = np.full(4, np.inf)  
    for idx in range(Quadrant_info.shape[0]):
        x_symbol = Quadrant_info[idx][0]
        y_symbol = Quadrant_info[idx][1]
        
        if 0 < x_symbol and y_symbol < 0:
            if distance2origin[idx] < Min_distance[0]:
                Min_distance[0] = copy.deepcopy(distance2origin[idx])
                Quadrant_idx[0] = copy.deepcopy(idx)

        elif x_symbol < 0 and y_symbol < 0:
            if distance2origin[idx] < Min_distance[1]:
                Min_distance[1] = copy.deepcopy(distance2origin[idx])
                Quadrant_idx[1] = copy.deepcopy(idx)
        
        elif x_symbol < 0 and 0 < y_symbol:
            if distance2origin[idx] < Min_distance[2]:
                Min_distance[2] = copy.deepcopy(distance2origin[idx])
                Quadrant_idx[2] = copy.deepcopy(idx)

        elif 0 < x_symbol and 0 < y_symbol:
            if distance2origin[idx] < Min_distance[3]:
                Min_distance[3] = copy.deepcopy(distance2origin[idx])
                Quadrant_idx[3] = copy.deepcopy(idx)
        else:
            print("label equal to center!")

    # ReturnValue = [points[int(Quadrant_idx[0])],
    #                points[int(Quadrant_idx[1])],
    #                points[int(Quadrant_idx[2])],
    #                points[int(Quadrant_idx[3])]]
    ReturnValue = [points[int(Quadrant_idx[0])],
                   points[int(Quadrant_idx[3])],
                   points[int(Quadrant_idx[2])],
                   points[int(Quadrant_idx[1])]]
    ReturnValue = np.array(ReturnValue)

    return ReturnValue


# 输入：另外一组二维平面点(n, 2), 求右上角四个红色区域点
# 输出：包围点A形成矩形的四个角点，从右上角开始，逆时针排序
def InferTableCorner(points):
    # 对x坐标从小到大排序
    l_points = points[np.argsort(points[:, 0])[0:2]]    # 左侧俩标签
    lt_point = l_points[np.argsort(l_points[:, 1])[0]]    # 左上角
    lb_point = l_points[np.argsort(l_points[:, 1])[1]]    # 左下角

    r_points = points[np.argsort(points[:, 0])[2:]]     # 右侧俩标签
    rt_point = r_points[np.argsort(r_points[:, 1])[0]]    # 右上角
    rb_point = r_points[np.argsort(r_points[:, 1])[1]]    # 右下角

    reruenPoints = np.array([rt_point, rb_point, lb_point, lt_point])
    return reruenPoints



# 给图像画标签
# 输入image, position(n*2)
def draw_circle(img, position):

    for idx in range(position.shape[0]):
        cv2.circle(img, position[idx], 15, (0,255,255), 2)
    
    return img

# 给图像画标签
# 输入image, position(n*2)
def draw_point(img, position):

    color = (0, 255, 0)  # 绿色 (BGR格式)  
    radius = 4  # 点的半径  
    thickness = -1  # 如果为负值，则填充点 

    for idx in range(position.shape[0]):
        cv2.circle(img, position[idx], radius, color, thickness) 

    return img

def detect_tabel(ProcessedImage, LabelsCenter, TableWorldCoor, K, cam2robot):
     



    # 2. 按照逻辑求标签像素坐标
    CurrentCorner = InferTableCorner(LabelsCenter)

    # 画点
    ProcessedImage = draw_point(ProcessedImage, CurrentCorner)

    # 3. 求四个角点在世界坐标系下的位置
    CornerInCam = np.empty((4, 4))
    # CurrentCorner_f = CurrentCorner.astype(np.float64)
    CurrentCorner_f = LabelsCenter.astype(np.float64)
    ret, R_pnp, T_pnp = cv2.solvePnP(TableWorldCoor, CurrentCorner_f, K, np.zeros(5))
    CornerInCam[:3, :3] = cv2.Rodrigues(R_pnp)[0]
    CornerInCam[:3, 3:] = T_pnp
    CornerInCam[3, :] = (0,0,0,1)

    # camera 2 robot
    CornerinRobot = cam2robot @ CornerInCam  # n,4,4


    # 4. 计算箱子相对与机器人的航向角
    # cal the angle between projection of grasp ry and robot ry
    y_dir_Conner = CornerinRobot[:3, :3] @ np.array([[0],[1],[0]]).flatten()
    YawShelf = np.arctan(-y_dir_Conner[0]/y_dir_Conner[1])*180/np.pi

    draw_arrow(ProcessedImage, CornerInCam[:3, :3], CornerInCam[:3, 3:], K, line_length=102)

    return CornerinRobot, YawShelf, CornerInCam


# 检测标签算法
# 输入：图像，货架中心点，货架在世界系的坐标，相机内参
def detect_label(image, center_point, ShelfWorldCoor, K, cam2robot):

    ##############################################
    # 调试使用的初始化方式
    # 货架在世界坐标系下的位置
    # ShelfWorldCoor = np.array([[0, 0, 0],[327, 0, 0],[327, 380, 0],[0, 380, 0]], float)
    # qr中心点坐标
    # center_point = np.array([345, 270])
    # 相机内参
    # K = np.array([[291.1, 0, 319.5],
    #                       [0, 291.07, 199.5],
    #                       [0, 0, 1]], dtype=np.float32)




    # 相机转机器人坐标系的齐次矩阵
    # cam2robot = np.array([
    #     [  0.02402378,  -0.50812134 ,  0.86095039 ,117.17293646],
    #     [ -0.99868218 , -0.05126587 , -0.00238941 , 20.205979  ],
    #     [  0.04535148 , -0.85975841 , -0.50868332, 214.46518944],
    #     [  0.         ,  0.        ,   0.        ,   1.        ]])
    ##############################################

    # 0.创建返回值
    CornerinRobot = np.eye(4)
    YawShelf = 0

    # 1. 对原图进行黄色标签检测
    # ProcessedImage, LabelMask, LabelsCenter = detect_yellow_rectangles(image)
    ProcessedImage, LabelMask, LabelsCenter = detect_red_shelf_rectangles_filtered(image)
    if LabelsCenter.shape[0] < 4:
        return CornerinRobot, YawShelf
    
    # 画标签
    ProcessedImage = draw_circle(ProcessedImage, LabelsCenter)

    # 2. 求离给定中心最近的四个角点的标签坐标
    CurrentCorner = InferCorner(center_point, LabelsCenter)

    # 画点
    ProcessedImage = draw_point(ProcessedImage, CurrentCorner)

    # 3. 求四个角点在世界坐标系下的位置
    CornerInCam = np.empty((4, 4))
    CurrentCorner_f = CurrentCorner.astype(np.float64)
    ret, R_pnp, T_pnp = cv2.solvePnP(ShelfWorldCoor, CurrentCorner_f, K, np.zeros(5))
    CornerInCam[:3, :3] = cv2.Rodrigues(R_pnp)[0]
    CornerInCam[:3, 3:] = T_pnp
    CornerInCam[3, :] = (0,0,0,1)

    # camera 2 robot
    CornerinRobot = cam2robot @ CornerInCam  # n,4,4


    # 4. 计算箱子相对与机器人的航向角
    # cal the angle between projection of grasp ry and robot ry
    y_dir_Conner = CornerinRobot[:3, :3] @ np.array([[0],[1],[0]]).flatten()
    YawShelf = np.arctan(-y_dir_Conner[0]/y_dir_Conner[1])*180/np.pi

    draw_arrow(ProcessedImage, CornerInCam[:3, :3], CornerInCam[:3, 3:], K, line_length=102)

    return CornerinRobot, YawShelf

def draw_polygon(img, Points, R, T, K, D=np.zeros(5), line_length=0.1, color=(0,0,255)):
    # 多边形顶点转到像素坐标系
    Points_pixel, _ = cv2.projectPoints(Points, R, T, K, D)
    Points_pixel = np.int32(Points_pixel).reshape(-1, 2)

    # 绘制一个闭合的多边形, True:是否封闭，2：线宽度
    cv2.polylines(img, [Points_pixel], True, color, 2)
    

class ManageSubscriber():
    def __init__(self):
        self.system_command = [1, 2, 2]  # perception switch is default off
        # 订阅行数列数[row, col]
        rospy.Subscriber("/perception/switch", Int32MultiArray, self.manage_callback)


    def manage_callback(self, data):
        self.system_command = data.data


class ObjectDection(ImageSubscriber, ManageSubscriber):
    def __init__(self):
        ImageSubscriber.__init__(self)
        ManageSubscriber.__init__(self)
        self.i_QRDetect = QRDetect()

        self.freq_sample = 20  # running frequency: 5 or 10 Hz is recommanded
        self.result_volume = 10
        self.timer = rospy.Timer(rospy.Duration(1/self.freq_sample), self.perception_callback)
        self.pub_img = rospy.Publisher('/perception/object_detection/display_img', Image, queue_size=self.freq_sample)
        self.pub_img_shelf = rospy.Publisher('/perception/object_detection/display_img_shelf', Image, queue_size=self.freq_sample)
        self.pub_img_table = rospy.Publisher('/perception/object_detection/display_img_table', Image, queue_size=self.freq_sample)
        self.pub_arm_pose = rospy.Publisher('/perception/object_detection/arm_pose', ArmPoseArray, queue_size=self.freq_sample)
        self.pub_table_pose = rospy.Publisher('/perception/object_detection/table_pose', ArmPoseArray, queue_size=self.freq_sample)

        # actural size (mm)
        self.qr_world = np.array([[0, 0, 0],[58.6, 0, 0],[58.6, 57, 0],[0, 57, 0]], float)
        self.box_world = np.array([[60,-157,190], [60,157,190], [60,157,-25], [60,-157,-25]], float)
        self.grasps_box_left = np.array([[-105,-160,0], [50, 50, 0]]) # draw carry points
        self.grasps_box_right = np.array([[-105,160,0], [50, -50, 0]]) # draw carry points
        # 货架在世界坐标系下的位置
        self.ShelfWorldCoor = np.array([[0, 0, 0],[380, 0, 0],[380, 205, 0],[0, 205, 0]], float)    # sf_shanghai


        
        # 货架在世界坐标系下的位置
        self.TableWorldLabel = np.array([[0, 0, 0],[0, 0, -200],[0, 449, -200],[0, 449, 0]], float)  # 桌子上标签的点

        markerSize = 277
        self.WC_TableQR = np.array([[0, 0, 0],[markerSize, 0, 0],[markerSize, markerSize, 0],[0, markerSize, 0]], float)       # 桌子上的QR码
        self.WC_TableQR = np.array([[0, 0, 0],[0, 0, markerSize],[0, markerSize, markerSize],[0, markerSize, 0]], float)       # 桌子上的QR码

        # Rmt Homo matrix
        self.cam2robot = np.array([[ 0.00195592,  -0.50377593,   0.86383215, 112.10644183 ],
                            [ -0.99967441,  -0.02296195,  -0.01112762,  14.7286837 ],
                            [  0.0254411,   -0.86352913,  -0.50365682, 220.34255919 ],
                            [  0.,           0.,           0.,           1.        ]])
        

        self.qr2grasp = np.eye(4)
        self.qr2grasp[:3,3] = 30.0, -80.0, 20.0

        self.qr2grasp_left = np.eye(4)
        self.qr2grasp_left[:3,3] = 30.0, 80.0, 20.0
        # height of each rows on shelf in robot coordinate system
        self.rows_z = np.array([[360], [110], [-300], [-600], [-850]], float)
        self.shelf_row_z = np.array([[360-150], [110-150], [-300-150], [-600-150], [-850-150]], float)
        self.grasp_row_z = np.array([[-99999], [54], [-326], [-646], [-99999]], float)
        # = np.array([[360 - 112], [110 - 112], [-300 - 112], [-600 - 112], [-850 - 112]], float)
        # -412

        # ma for smoothing result
        self.clean_buffer()

        print('%s initiated...' % self.__class__.__name__)

    def clean_buffer(self):
        self.lift_xyz_buffer = deque(maxlen=self.result_volume)
        self.lift_xyz_buffer_left = deque(maxlen=self.result_volume)        
        self.lift_rz_buffer = deque(maxlen=self.result_volume)
        self.carry_xyz_buffer_left = deque(maxlen=self.result_volume)
        self.carry_xyz_buffer_right = deque(maxlen=self.result_volume)
        self.carry_rz_buffer = deque(maxlen=self.result_volume)
        self.shelf_xyz_buffer = deque(maxlen=self.result_volume)
        self.shelf_rz_buffer = deque(maxlen=self.result_volume)
        self.box_pull_buffer = deque(maxlen=self.result_volume)
        self.table_xyz_buffer = deque(maxlen=self.result_volume)
        self.table_rz_buffer = deque(maxlen=self.result_volume)





    


    # @log
    def perception_callback(self, event):
        # wont run if perception switch is turned off
        if not self.system_command[0]:
            print("Alg is shut down!")
            self.clean_buffer()
            return

        try:
            K = np.array(self.K).reshape(3,3)
            rgb = self.rgb.copy()
        except AttributeError as e:
            print(e)
            return

        # initiate grasp, default t matrix is inf 
        grasps = [
            Grasp(order='xyz', Tm=np.ones([3])*float('+inf')), 
            Grasp(order='xyz', Tm=np.ones([3])*float('+inf')),
            Grasp(order='xyz', Tm=np.ones([3])*float('+inf')), 
            Grasp(order='xyz', Tm=np.ones([3])*float('+inf')),
            Grasp(order='xyz', Tm=np.ones([3])*float('+inf')),
            Grasp(order='xyz', Tm=np.ones([3])*float('+inf')),
            Grasp(order='xyz', Tm=np.ones([3])*float('+inf'))]


        # 拷贝文件
        rgb_table = np.copy(rgb)


####################
        # ######################## table detect

        # # table detection
        # # 1. 对原图进行黄色标签检测
        # ProcessedImage1, LabelMask1, LabelsCenter1 = detect_red_table_rectangles_filtered(rgb_table)
        # # 神经网络计算
        # # wws: 1432
        # # wt: 4123

        # # 画标签
        # ProcessedImage1 = draw_circle(rgb_table, LabelsCenter1)
        # if LabelsCenter1.shape[0] == 4:
        #     # 2. 将四个标签检测出来
        #     CornerInRobot1, YawShelf1, CornerInCamera1 = detect_tabel(ProcessedImage1, LabelsCenter1, self.TableWorldLabel, K, self.cam2robot)
        
        #     # 3. 赋值           
        #     self.table_xyz_buffer.append(CornerInRobot1[:3, 3:])
        #     self.table_rz_buffer.append(YawShelf1)
        #     if len(self.table_rz_buffer)==self.result_volume:
        #         grasps[6] = Grasp(order='xyz', R=[0,0,np.mean(self.table_rz_buffer)])
        #         grasps[6].trans = np.mean(self.table_xyz_buffer, axis=0)

        #     # 4. 根据标签，把桌子画出来把桌子所在区域画在二维图像上
        #     # axis_points = np.float32([[0, 0, 0], [line_length, 0, 0], [0, line_length, 0], [0, 0, line_length]])
        #     Table_w = 449
        #     Table_h = 800
        #     TableWorldCorner = np.float32([[0, 0, 0],[0, 0, -Table_w],[0, Table_h, -Table_w],[0, Table_h, 0]]) # 桌子四个角的角点在桌子坐标系
        #     TableWorldCorner = np.float32([[0, 0, 0],[0, 0, -Table_h],[0, Table_w, -Table_h],[0, Table_w, 0]]) # 桌子四个角的角点在桌子坐标系,从右上角，逆时针排序

        #     # 画多边形
        #     draw_polygon(ProcessedImage1, TableWorldCorner, CornerInCamera1[:3, :3], CornerInCamera1[:3, 3:], K, D=np.zeros(5), line_length=0.1, color=(0,0,255))
        # ########################
####################
        # box detection
        start_time = time.time()
        # yolo_output = yolo(rgb, conf=0.65, iou=0.2, verbose=False, half=True)[0]
        yolo_output = yolo(rgb, conf=0.45, iou=0.5, verbose=False, half=True)[0]
        # yolo_output = yolo_tmp(rgb, conf=0.45, iou=0.5, verbose=False, half=True)[0]


        # shelf detection
        rgb_shelf = np.copy(rgb)
        if len(yolo_output) > 0:
            print("run alg")
            boxes = yolo_output.boxes.xywh.cpu().numpy()
            cls_names = np.array([yolo_output.names[ici] for ici in yolo_output.boxes.cls.int().cpu().numpy()])
            kpts = yolo_output.keypoints.xy.cpu().numpy()
            if len(cls_names) == len(kpts):
                # idx_conf = (yolo_output.keypoints.conf.cpu().numpy() > 0.8).all(axis=1)  # n,4 -> n
                idx_conf = (yolo_output.keypoints.conf.cpu().numpy() > 0.45).all(axis=1)  # n,4 -> n

                kpts = kpts[idx_conf]
                cls_names = cls_names[idx_conf]
                boxes = boxes[idx_conf]



                qr_kpts = kpts[cls_names=='qr']
                box_kpts = kpts[cls_names=='box']
                shelfRed_kpts = kpts[cls_names=='red_grid']
                shelfGreen_kpts = kpts[cls_names=='green_grid']

                # shelf detection（神经网络版本）
                row_assign, col_assign = self.system_command[1:]
                row_shelf= 2
                # row_assign = 2
                # col_assign = 0
                # 如果系统发的是偶数列（0开始数）是偶数，且红色结果数大于0
                if (col_assign%2 == 0) and 0 < len(shelfRed_kpts):
###########################
                    num_world = shelfRed_kpts.shape[0]
                    world_in_cam = np.empty((num_world, 4, 4))
                    # 计算6d位姿（坐标系），并转为RT矩阵
                    # Using 4 corners of kpt to solve PnP and get 6dof of kpt in camera coordinate system: RT
                    for idx, kpt in enumerate(shelfRed_kpts):                            
                        ret, R_vector, T_vector = cv2.solvePnP(self.ShelfWorldCoor, kpt, K, np.zeros(5))
                        world_in_cam[idx, :3, :3] = cv2.Rodrigues(R_vector)[0]
                        world_in_cam[idx, :3, 3:] = T_vector
                        world_in_cam[idx, 3, :] = (0,0,0,1)
                    
                    # 2. camera 2 robot
                    world_in_robot = self.cam2robot @ world_in_cam  # 输入输出都是:n,4,4 = 4,4 * n,4,4
                    world_in_robot[:, 1, 3] = world_in_robot[:, 1, 3] + 90
                    pose_in_robot = world_in_robot
###########################
                    # cal row on
                    pose_z = pose_in_robot[:, 2, 3]  # n,1: z value of each qr
                    pose_y = pose_in_robot[:, 1, 3]

                    pose_rowno = np.argmin(np.abs(pose_z - self.shelf_row_z), axis=0)  # n,1 - n,1 -> n

                    if np.isin(row_shelf, pose_rowno):
                        row_index = np.where(pose_rowno == row_shelf)

                        ### 画图
                        obj_id = 0
                        kps_uv_centres = np.mean(shelfRed_kpts, axis=1).astype(int)  # n,4,2 -> n,2
                        for _idx, _kpt in enumerate(shelfRed_kpts):
                            draw_arrow(rgb, world_in_cam[_idx, :3, :3], world_in_cam[_idx, :3, 3:], K, line_length=62)
                            # cv2.putText(rgb, f"{pose_rowno[_idx]}", kps_uv_centres[_idx], 2, 1, (255,0,255), 2, cv2.LINE_AA)

                            cv2.putText(rgb, f"{int(pose_y[obj_id])}", kps_uv_centres[_idx], 2, 1, (255,0,255), 2, cv2.LINE_AA)
                            obj_id = obj_id + 1

                        # cal col in
                        pose_in_ros_assign = pose_in_robot[row_index]
                        pose_y = pose_in_ros_assign[:, 1, 3]
                        yz_close_idx = np.argmin(np.abs(pose_y)) 
                        pose_assign = pose_in_ros_assign[yz_close_idx]

                        y_dir_Conner = -pose_assign[:3, :3] @ np.array([[0],[0],[1]]).flatten()
                        YawShelf = np.arctan(y_dir_Conner[1]/y_dir_Conner[0])*180/np.pi

                        pose_assign_inCam = world_in_cam[yz_close_idx]
                        draw_arrow(rgb, pose_assign_inCam[:3, :3], pose_assign_inCam[:3, 3:], K, line_length=102)

                        # 待发送
                        pose_assign[1, 3] = pose_assign[1, 3] - 90
                        self.shelf_xyz_buffer.append(pose_assign[:3, 3:])
                        self.shelf_rz_buffer.append(YawShelf)
                        if len(self.shelf_rz_buffer)==self.result_volume:
                            grasps[4] = Grasp(order='xyz', R=[0,0,np.mean(self.shelf_rz_buffer)])
                            grasps[4].trans = np.mean(self.shelf_xyz_buffer, axis=0)

                if (col_assign%2 != 0) and 0 < len(shelfGreen_kpts):
###########################
                    num_world = shelfGreen_kpts.shape[0]
                    world_in_cam = np.empty((num_world, 4, 4))
                    # 计算6d位姿（坐标系），并转为RT矩阵
                    # Using 4 corners of kpt to solve PnP and get 6dof of kpt in camera coordinate system: RT
                    for idx, kpt in enumerate(shelfGreen_kpts):
                        ret, R_vector, T_vector = cv2.solvePnP(self.ShelfWorldCoor, kpt, K, np.zeros(5))                        
                        world_in_cam[idx, :3, :3] = cv2.Rodrigues(R_vector)[0]
                        world_in_cam[idx, :3, 3:] = T_vector
                        world_in_cam[idx, 3, :] = (0,0,0,1)
                    
                    # 2. camera 2 robot
                    world_in_robot = self.cam2robot @ world_in_cam  # 输入输出都是:n,4,4 = 4,4 * n,4,4
                    world_in_robot[:, 1, 3] = world_in_robot[:, 1, 3] + 90

                    pose_in_robot = world_in_robot
###########################
                    # cal row on
                    pose_z = pose_in_robot[:, 2, 3]  # n,1: z value of each qr
                    pose_rowno = np.argmin(np.abs(pose_z - self.shelf_row_z), axis=0)  # n,1 - n,1 -> n
                    pose_y = pose_in_robot[:, 1, 3]

                    if np.isin(row_shelf, pose_rowno):
                        row_index = np.where(pose_rowno == row_shelf)
                        obj_id = 0
                        ### 画图
                        kps_uv_centres = np.mean(shelfGreen_kpts, axis=1).astype(int)  # n,4,2 -> n,2
                        for _idx, _kpt in enumerate(shelfGreen_kpts):
                            draw_arrow(rgb, world_in_cam[_idx, :3, :3], world_in_cam[_idx, :3, 3:], K, line_length=62)
                            # cv2.putText(rgb, f"{pose_rowno[_idx]}", kps_uv_centres[_idx], 2, 1, (255,0,255), 2, cv2.LINE_AA)

                            cv2.putText(rgb, f"{int(pose_y[obj_id])}", kps_uv_centres[_idx], 2, 1, (255,0,255), 2, cv2.LINE_AA)
                            obj_id = obj_id + 1


                        # cal col in
                        pose_in_ros_assign = pose_in_robot[row_index]
                        pose_y = pose_in_ros_assign[:, 1, 3]
                        yz_close_idx = np.argmin(np.abs(pose_y)) 
                        pose_assign = pose_in_ros_assign[yz_close_idx]

                        y_dir_Conner = -pose_assign[:3, :3] @ np.array([[0],[0],[1]]).flatten()
                        YawShelf = np.arctan(y_dir_Conner[1]/y_dir_Conner[0])*180/np.pi

                        pose_assign_inCam = world_in_cam[yz_close_idx]
                        draw_arrow(rgb, pose_assign_inCam[:3, :3], pose_assign_inCam[:3, 3:], K, line_length=102)
                        
                        # 待发送
                        pose_assign[1, 3] = pose_assign[1, 3] - 90
                        self.shelf_xyz_buffer.append(pose_assign[:3, 3:])
                        self.shelf_rz_buffer.append(YawShelf)
                        if len(self.shelf_rz_buffer)==self.result_volume:
                            grasps[4] = Grasp(order='xyz', R=[0,0,np.mean(self.shelf_rz_buffer)])
                            grasps[4].trans = np.mean(self.shelf_xyz_buffer, axis=0)



                if len(qr_kpts) > 0:
                    # Using 4 corners of qr to solve PnP and get 6dof of qr in camera coordinate system
                    Rmt_pnp_all = np.empty((len(qr_kpts), 4, 4))
                    for _idx, _kpt in enumerate(qr_kpts):
                        ret, Rm_pnp, t_pnp = cv2.solvePnP(self.qr_world, _kpt, K, np.zeros(5))
                        Rmt_pnp_all[_idx, :3, :3] = cv2.Rodrigues(Rm_pnp)[0]
                        Rmt_pnp_all[_idx, :3, 3:] = t_pnp
                        Rmt_pnp_all[_idx, 3, :] = (0,0,0,1)
                    qr_robot = self.cam2robot @ Rmt_pnp_all  # n,4,4

                    # using acctural height of each row in shelf to identify the row number of each box
                    qr_robot_z = qr_robot[:, 2, 3]  # n,1: z value of each qr
                    qr_robot_rowno = np.argmin(np.abs(qr_robot_z - self.rows_z), axis=0)  # n,1 - n,1 -> n
                    qr_uv_centres = np.mean(qr_kpts, axis=1).astype(int)  # n,4,2 -> n,2
                    # draw xyz and row number of each qr on image
                    for _idx, _kpt in enumerate(qr_kpts):
                        draw_arrow(rgb, Rmt_pnp_all[_idx, :3, :3], Rmt_pnp_all[_idx, :3, 3:], K, line_length=62)
                        cv2.putText(rgb, f"{qr_robot_rowno[_idx]}", qr_uv_centres[_idx], 2, 1, (0,255,255), 2, cv2.LINE_AA)

                    # find the mid qr of each row 
                    qr_robot_y = qr_robot[:, 1, 3]  # n,1: y value of each qr
                    qridx_mid_in_row = []  # record the mid qr index of each row
                    qr_irange = np.arange(len(qr_robot_rowno))  # the index of each qr
                    # iterate each row
                    for rowno in range(len(self.rows_z)):
                        if sum(qr_robot_rowno==rowno) > 0:
                            mid_idx = np.argmin(np.abs(qr_robot_y[qr_robot_rowno==rowno]))
                            qridx_mid_in_row.append(qr_irange[qr_robot_rowno==rowno][mid_idx])
                            cv2.circle(rgb, qr_uv_centres[qridx_mid_in_row[-1]], 15, (0,255,0), 2)
                        else:
                            # no qr found in this row, skip
                            qridx_mid_in_row.append(-999)

                    # the boxes in the 3th row is what we want to grasp
                    # processing the final result
                    if qridx_mid_in_row[row_assign] != -999:
                        idx = qridx_mid_in_row[row_assign]
                        ##########
                        # # shelf detection（颜色标签版本）
                        # center_qr = qr_kpts[idx]
                        # center_point = np.mean(center_qr, axis=0)
                        # CornerInRobot, YawShelf = detect_label(rgb_shelf, center_point, self.ShelfWorldCoor, K, self.cam2robot)
                        # self.shelf_xyz_buffer.append(CornerInRobot[:3, 3:])
                        # self.shelf_rz_buffer.append(YawShelf)
                        # if len(self.shelf_rz_buffer)==self.result_volume:
                        #     grasps[4] = Grasp(order='xyz', R=[0,0,np.mean(self.shelf_rz_buffer)])
                        #     grasps[4].trans = np.mean(self.shelf_xyz_buffer, axis=0)
                        ##########
                        cv2.circle(rgb, qr_uv_centres[idx], 20, (255,0,0), 3)
                        grasp_cam = Rmt_pnp_all[idx] @ self.qr2grasp
                        grasp_cam_left = Rmt_pnp_all[idx] @ self.qr2grasp_left
                        draw_arrow(rgb, grasp_cam[:3, :3], grasp_cam[:3, 3:], K, line_length=102)
                        draw_arrow(rgb, grasp_cam_left[:3, :3], grasp_cam_left[:3, 3:], K, line_length=102)

                        lift_robot = self.cam2robot @ grasp_cam
                        lift_robot_left = self.cam2robot @ grasp_cam_left

                        # 计算右钩取点
                        # cal the angle between projection of grasp ry and robot ry
                        unit_y_robot = lift_robot[:3, :3] @ np.array([[0],[1],[0]]).flatten()
                        lift_rz = np.arctan(-unit_y_robot[0]/unit_y_robot[1])*180/np.pi
                        # print(unit_y_robot[0], "      ", unit_y_robot[1])
                        # print("lift_rz: ", lift_rz)
                        self.lift_xyz_buffer.append(lift_robot[:3, 3:])
                        self.lift_rz_buffer.append(lift_rz)
                        if len(self.lift_rz_buffer)==self.result_volume:
                            grasps[0] = Grasp(order='xyz', R=[0,0,np.mean(self.lift_rz_buffer)])
                            grasps[0].trans = np.mean(self.lift_xyz_buffer, axis=0)
                            # tmp adj
                            grasps[0].trans[2] = self.grasp_row_z[row_assign] # -326.0

                        # 计算左钩取点, 旋转不变，只改变位置
                        self.lift_xyz_buffer_left.append(lift_robot_left[:3, 3:])
                        if len(self.lift_xyz_buffer_left)==self.result_volume:
                            grasps[1] = Grasp(order='xyz', R=[0,0,np.mean(self.lift_rz_buffer)])
                            grasps[1].trans = np.mean(self.lift_xyz_buffer_left, axis=0)
                            # tmp adj
                            grasps[1].trans[2] = self.grasp_row_z[row_assign] # -326.0

                if len(box_kpts) > 0:
                    box_size = [xywh[2]*xywh[3] for xywh in boxes[cls_names=='box']]
                    closest_box_kpts = box_kpts[np.argmax(box_size)]
                    for _p in closest_box_kpts.astype(int):
                        cv2.circle(rgb, _p, 3, (0,0,255),-1)
                    Rmt_pnp_box = np.eye(4)
                    ret, Rm_pnp, t_pnp = cv2.solvePnP(self.box_world, closest_box_kpts, K, np.zeros(5))
                    Rmt_pnp_box[:3,:3] = cv2.Rodrigues(Rm_pnp)[0]
                    Rmt_pnp_box[:3,3:] = t_pnp
                    draw_arrow(rgb, Rmt_pnp_box[:3,:3], Rmt_pnp_box[:3,3:], K, line_length=162)
                    
                    _g = np.eye(4)
                    _g[:3,3] = self.grasps_box_left[0]
                    _g = Rmt_pnp_box @ _g
                    draw_grasp(rgb, _g[:3, :3], _g[:3, 3:], self.grasps_box_left[1], K)  # draw carry points
                    carry_robot_left = self.cam2robot @ _g
                    _g = np.eye(4)
                    _g[:3,3] = self.grasps_box_right[0]
                    _g = Rmt_pnp_box @ _g
                    draw_grasp(rgb, _g[:3, :3], _g[:3, 3:], self.grasps_box_right[1], K)  # draw carry points
                    carry_robot_right = self.cam2robot @ _g
                    unit_y_robot = carry_robot_left[:3, :3] @ np.array([[0],[1],[0]]).flatten()
                    carry_rz = np.arctan(-unit_y_robot[0]/unit_y_robot[1])*180/np.pi

                    self.carry_xyz_buffer_left.append(carry_robot_left[:3, 3:])
                    self.carry_xyz_buffer_right.append(carry_robot_right[:3, 3:])
                    self.carry_rz_buffer.append(carry_rz)
                    if len(self.carry_rz_buffer)==self.result_volume:
                        grasps[2] = Grasp(order='xyz', R=[0,0,np.mean(self.carry_rz_buffer)])
                        grasps[2].trans = np.mean(self.carry_xyz_buffer_left, axis=0)
                        grasps[3] = Grasp(order='xyz', R=[0,0,np.mean(self.carry_rz_buffer)])
                        grasps[3].trans = np.mean(self.carry_xyz_buffer_right, axis=0)
        table_qr = True
        # if len(yolo_output) > 0 and not(table_qr):
        #     boxes = yolo_output.boxes.xywh.cpu().numpy()
        #     cls_names = np.array([yolo_output.names[ici] for ici in yolo_output.boxes.cls.int().cpu().numpy()])
        #     kpts = yolo_output.keypoints.xy.cpu().numpy()
        #     if len(cls_names) == len(kpts):
        #         # idx_conf = (yolo_output.keypoints.conf.cpu().numpy() > 0.8).all(axis=1)  # n,4 -> n
        #         idx_conf = (yolo_output.keypoints.conf.cpu().numpy() > 0.45).all(axis=1)  # n,4 -> n

        #         kpts = kpts[idx_conf]
        #         cls_names = cls_names[idx_conf]
        #         boxes = boxes[idx_conf]



        #         qr_kpts = kpts[cls_names=='qr']
        #         box_kpts = kpts[cls_names=='box']
        #         table_kpts = kpts[cls_names=='table_point']

        #         if len(0 < table_kpts):
        #             ######################## table detect

        #             # table detection
        #             # 1. 对原图进行黄色标签检测
        #             # ProcessedImage1, LabelMask1, LabelsCenter1 = detect_red_table_rectangles_filtered(rgb_table)
        #             # 神经网络计算
        #             # wws: 1432
        #             # wt: 4123
        #             if len(table_kpts) == 1:
        #                 new_idx = [1, 0, 3, 2]
        #                 LabelsCenter1 = table_kpts[0][new_idx].astype(np.int32)
        #                 # 画标签
        #                 ProcessedImage1 = draw_circle(rgb_table, LabelsCenter1)
        #                 if LabelsCenter1.shape[0] == 4:
        #                     # 2. 将四个标签检测出来
        #                     CornerInRobot1, YawShelf1, CornerInCamera1 = detect_tabel(ProcessedImage1, LabelsCenter1, self.TableWorldLabel, K, self.cam2robot)
                        
        #                     # 3. 赋值           
        #                     self.table_xyz_buffer.append(CornerInRobot1[:3, 3:])
        #                     self.table_rz_buffer.append(YawShelf1)
        #                     if len(self.table_rz_buffer)==self.result_volume:
        #                         grasps[6] = Grasp(order='xyz', R=[0,0,np.mean(self.table_rz_buffer)])
        #                         grasps[6].trans = np.mean(self.table_xyz_buffer, axis=0)

        #                     # 4. 根据标签，把桌子画出来把桌子所在区域画在二维图像上
        #                     # axis_points = np.float32([[0, 0, 0], [line_length, 0, 0], [0, line_length, 0], [0, 0, line_length]])
        #                     Table_w = 449
        #                     Table_h = 800
        #                     TableWorldCorner = np.float32([[0, 0, 0],[0, 0, -Table_w],[0, Table_h, -Table_w],[0, Table_h, 0]]) # 桌子四个角的角点在桌子坐标系
        #                     TableWorldCorner = np.float32([[0, 0, 0],[0, 0, -Table_h],[0, Table_w, -Table_h],[0, Table_w, 0]]) # 桌子四个角的角点在桌子坐标系,从右上角，逆时针排序

        #                     # 画多边形
        #                     draw_polygon(ProcessedImage1, TableWorldCorner, CornerInCamera1[:3, :3], CornerInCamera1[:3, 3:], K, D=np.zeros(5), line_length=0.1, color=(0,0,255))
                        ########################
        if table_qr:
            LabelsCenter1 = self.i_QRDetect.qr_detect(rgb)
            if LabelsCenter1 is not None:
            
                # 画标签
                ProcessedImage1 = draw_circle(rgb_table, LabelsCenter1)

                if LabelsCenter1.shape[0] == 4:
                    # 2. 将四个标签检测出来
                    CornerInRobot1, YawShelf1, CornerInCamera1 = detect_tabel(ProcessedImage1, LabelsCenter1, self.WC_TableQR, K, self.cam2robot)

                    self.table_xyz_buffer.append(CornerInRobot1[:3, 3:])
                    self.table_rz_buffer.append(YawShelf1)
                    if len(self.table_rz_buffer)==self.result_volume:
                        grasps[6] = Grasp(order='xyz', R=[0,0,np.mean(self.table_rz_buffer)])
                        grasps[6].trans = np.mean(self.table_xyz_buffer, axis=0)


        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"执行时间: {elapsed_time} 秒")
##################
        
##################
        ##########
        grasps[5] = Grasp(order='xyz', R=[0,0,0])
        pull_distance = float(grasps[4].trans[0] - 250 - (grasps[0].trans[0]-30)) -20
        grasps[5].trans = np.array([pull_distance, 0, 0])
        ##########

        ##########
        # 2024.12.23临时调试  调完注释
        # 勾取点设置固定值
        # grasps[0].rot.R = np.array([0, 0, -0.859752])
        # grasps[0].trans = np.array([629.773, -130.752, -646])

        # # 拉取距离设置固定值
        # grasps[5].trans = np.array([190, 0, 0])
        ##########
        print('right hook euler angle: ', grasps[0].rot.R)
        print('right hook 3d position: ', grasps[0].trans.flatten().astype(int).tolist())
        print('left hook euler angle: ', grasps[1].rot.R)
        print('left hook 3d position: ', grasps[1].trans.flatten().astype(int).tolist())
        print('carry left euler angle: ', grasps[2].rot.R)
        print('carry left 3d position: ', grasps[2].trans.flatten().astype(int).tolist())
        print('carry right euler angle: ', grasps[3].rot.R)
        print('carry right 3d position: ', grasps[3].trans.flatten().astype(int).tolist())
        print('shelf euler angle: ', grasps[4].rot.R)
        print('shelf right 3d position: ', grasps[4].trans.flatten().astype(int).tolist())

        print('pull distance: ', pull_distance)
        # output
        ap1 = ArmPose(gesture='lift', left=grasps[1].to_msg(), right=grasps[0].to_msg())
        ap2 = ArmPose(gesture='carry', left=grasps[2].to_msg(), right=grasps[3].to_msg())
        ap3 = ArmPose(gesture='shelf', right=grasps[4].to_msg())
        ap4 = ArmPose(gesture='pull_distance', right=grasps[5].to_msg())
        ap5 = ArmPose(gesture='table_pose', right=grasps[6].to_msg())


        self.pub_arm_pose.publish(ArmPoseArray((ap1, ap2, ap3, ap4)))
        if math.isinf(grasps[6].trans[0]) or math.isinf(grasps[6].trans[1]) or math.isinf(grasps[6].trans[2]):
            print('table pose error ')
            pass
        else:
            print('table euler angle: ', grasps[6].rot.R)
            print('table right 3d position: ', grasps[6].trans.flatten().astype(int).tolist())
            self.pub_table_pose.publish(ArmPoseArray((ap5,)))

        self.pub_img.publish(self.bridge.cv2_to_imgmsg(rgb, 'bgr8'))
        # self.pub_img_shelf.publish(self.bridge.cv2_to_imgmsg(rgb_shelf, 'bgr8'))
        self.pub_img_table.publish(self.bridge.cv2_to_imgmsg(rgb_table, 'bgr8'))
        cv2.imwrite(CODE_DIR + 'box_det.jpg', rgb)
        # cv2.imwrite(CODE_DIR + 'shelf_det.jpg', rgb_shelf)



if __name__ == '__main__':
    ASSETS = '/home/kepler/catkin_ws/src/perception/src/assets/'


    yolo = YOLO(ASSETS + 'yolov8s-pose_shanghai_v2.pt') # 


    
    
    rospy.init_node('object_detection', anonymous=False)
    object_dection = ObjectDection()
    rospy.spin()
