import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import open3d as o3d
from std_msgs.msg import String, Int32
from geometry_msgs.msg import Pose, Point, Quaternion
import numpy as np



class YoloImageSubscriber(Node):
    def __init__(self):
        super().__init__('yolo_image_subscriber')

        # 初始化YOLO模型
        # self.model = YOLO("/home/lz/Robotic_vision_kepler/ObjectDection/runs/detect/train3/weights/best.pt")
        self.model = YOLO("/home/kepler/Spatial_AI/assets/best20250225.pt")

        # 创建图像订阅者
        self.image_subscription = self.create_subscription(Image, '/camera/color/image_raw', self.rgb_callback, 10)

        # 订阅触发信号
        self.subscription_start_navigation = self.create_subscription(Int32, '/start_navigation', self.start_navigation_callback, 10)

        # 创建深度图订阅者
        self.depth_subscription = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        # 订阅相机内参
        self.create_subscription(CameraInfo, '/camera/rgb/camera_info', self.camera_info_callback, 10)

        # 创建信号灯状态话题发布器
        self.publisher_tricolor_status = self.create_publisher(String, '/tricolor_status_topic', 10)

        # 创建按钮位姿话题发布器
        self.pose_pub = self.create_publisher(Pose, '/button_target_pose', 10)

        # 创建cv_bridge实例
        self.bridge = CvBridge()

        self.rgb_image = None
        self.depth_image = None

        self.detection = False

    def start_navigation_callback(self, msg):
        # 回调函数处理接收到的消息
        self.detection = msg.data
        self.get_logger().info(f'Received navigation start signal: {msg.data}')

    def rgb_callback(self, msg):
        """处理 RGB 图像并执行 YOLOv8 推理"""
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        if self.depth_image is not None:
            self.process_images()

    def depth_callback(self, msg):
        """处理深度图像"""
        # 将 ROS 图像消息转换为 OpenCV 图像
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")  # 深度图通常为 16 位无符号整数
        if self.rgb_image is not None:
            self.process_images()

    def camera_info_callback(self, msg):
        """处理相机内参"""
        self.camera_info = msg
        # 提取内参
        self.fx = self.camera_info.k[0]  # 焦距 fx
        self.fy = self.camera_info.k[4]  # 焦距 fy
        self.cx = self.camera_info.k[2]  # 主点 cx
        self.cy = self.camera_info.k[5]  # 主点 cy

        if self.rgb_image is not None and self.depth_image is not None:
            self.process_images()

    def process_images(self):
        """将 RGB 图像和深度图像结合，进行目标检测，并计算质心位置"""
        # 进行 YOLOv8 推理
        if self.detection != 1:
            return
        results = self.model(self.rgb_image, conf=0.7)

        for result in results[0].boxes:  # 获取检测框
            # Ensure we have enough values
            if len(result.xywh[0]) == 4:
                x_center, y_center, width, height = result.xywh[0]  # Unpack coordinates only
                confidence = result.conf[0]  # Confidence score
                class_id = result.cls[0]  # Class ID

                if 1 <= class_id <= 3:
                    class_name = results[0].names[int(class_id)]  # 类别名称
                    # 发布信号灯状态信息
                    msg = String()
                    # 循环发送红、黄、绿状态
                    # states = ['Red', 'Yellow', 'Green']
                    msg.data = class_name
                    self.publisher_tricolor_status.publish(msg)
                    self.get_logger().info(f'Publishing: "{msg.data}"')
                else:
                    # 获取对应类别的名称
                    class_name = results[0].names[int(class_id)]

                    # 将检测框映射到深度图
                    x_min = int(x_center - width / 2)
                    x_max = int(x_center + width / 2)
                    y_min = int(y_center - height / 2)
                    y_max = int(y_center + height / 2)

                    # 获取深度图对应区域的深度信息
                    depth_region = self.depth_image[y_min:y_max, x_min:x_max]

                    # 计算目标区域的 3D 点云质心位置
                    target_points = self.get_3d_points(depth_region, x_min, y_min)
                    centroid = self.calculate_centroid(target_points)

                    # 发布位姿
                    self.publish_pose(centroid)
            else:
                # If unexpected result format, handle appropriately
                x_center, y_center, width, height, confidence, class_id = result.xywh[0]

    def get_3d_points(self, depth_region, x_offset, y_offset):
        """从深度图像区域获取 3D 点云"""
        points = []
        for v in range(depth_region.shape[0]):
            for u in range(depth_region.shape[1]):
                depth_value = depth_region[v, u]
                if depth_value == 0:  # 跳过无效深度
                    continue
                z = depth_value / 1000.0  # 如果深度图是以毫米为单位，则转换为米
                x = (u + x_offset - self.cx) * z / self.fx  # 使用相机内参 fx 和 cx
                y = (v + y_offset - self.cy) * z / self.fy  # 使用相机内参 fy 和 cy
                points.append([x, y, z])
        return np.array(points)

    def calculate_centroid(self, points):
        """计算点云的质心"""
        if points.shape[0] == 0:
            return Point(0, 0, 0)
        centroid = np.mean(points, axis=0)
        return Point(centroid[0], centroid[1], centroid[2])

    def publish_pose(self, centroid):
        """发布目标的位姿"""
        pose_msg = Pose()
        pose_msg.position = centroid

        # 假设目标朝向为零旋转
        pose_msg.orientation = Quaternion(0, 0, 0, 1)

        self.pose_pub.publish(pose_msg)
        self.get_logger().info(f"Published pose: {pose_msg.position}")

def main(args=None):
    rclpy.init(args=args)

    yolo_image_subscriber = YoloImageSubscriber()
    rclpy.spin(yolo_image_subscriber)

    yolo_image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
