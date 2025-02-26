import stag
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import threading
import json
import os
import time

class RGBDCameraSubscriber(Node):
    def __init__(self, rgb_topic, camera_info_topic):
        super().__init__('rgbd_camera_subscriber')

        self.rgb_topic = rgb_topic
        self.camera_info_topic = camera_info_topic

        self.rgb = None  # Placeholder for the latest RGB frame
        self.mtx = None  # Camera intrinsic matrix
        self.dist = None  # Camera distortion coefficients
        self.target_pose = None  # Target pose for mode 1

        self.bridge = CvBridge()

        self.mode = 2  # Default mode (1: fixed target, 2: random target)

        # Transformation matrix from camera to robot coordinate system
        self.cam2robot = np.array([
            [-0.03075, -0.48789, 0.87236, 135.94691],
            [-0.99888, 0.04628, -0.00932, 22.09545],
            [-0.03583, -0.87168, -0.48877, 241.49433],
            [0.0, 0.0, 0.0, 1.0]
        ])

        self.config_file = "target_pose.json"
        self.load_target_pose()

        # Create subscribers
        self.rgb_subscriber = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            10
        )

        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10
        )

        # Publisher for Pose
        self.pose_publisher = self.create_publisher(Pose, 'Marker_pose', 10)

        # Publisher for Object Z Position
        self.object_publisher = self.create_publisher(Pose, '/bmw/put_down_pos', 10)

        # Thread for spinning the node
        self.spin_thread = threading.Thread(target=self.spin_node)
        self.spin_thread.daemon = True
        self.spin_thread.start()

    def rgb_callback(self, msg):
        try:
            self.rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error converting RGB image: {e}")

    def camera_info_callback(self, msg):
        if self.mtx is None and self.dist is None:
            self.mtx = np.array(msg.k).reshape(3, 3)
            self.dist = np.array(msg.d)
            self.get_logger().info("Camera intrinsics and distortion coefficients updated.")

    def publish_pose(self, position, orientation):
        pose_msg = Pose()
        pose_msg.position.x = position[0]/1000
        pose_msg.position.y = position[1]/1000
        pose_msg.position.z = position[2]/1000
        pose_msg.orientation.x = orientation[0]
        pose_msg.orientation.y = orientation[1]
        pose_msg.orientation.z = orientation[2]
        pose_msg.orientation.w = orientation[3]
        time.sleep(1)
        self.pose_publisher.publish(pose_msg)

    def publish_object_pose(self, position, orientation):
        pose_msg = Pose()
        pose_msg.position.x = position[0]
        pose_msg.position.y = position[1]
        pose_msg.position.z = position[2]
        pose_msg.orientation.x = orientation[0]
        pose_msg.orientation.y = orientation[1]
        pose_msg.orientation.z = orientation[2]
        pose_msg.orientation.w = orientation[3]
        self.object_publisher.publish(pose_msg)

    def calculate_pose(self, rvec, tvec):
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Combine rotation matrix and translation vector into a transformation matrix
        transform_camera = np.eye(4)
        transform_camera[:3, :3] = rotation_matrix
        transform_camera[:3, 3] = tvec.flatten()

        # Transform to robot coordinate system
        # transform_robot = np.dot(self.cam2robot, transform_camera)
        transform_robot = transform_camera

        position = transform_robot[:3, 3]

        # Extract orientation as a quaternion
        rotation = transform_robot[:3, :3]
        quaternion = self.rotation_matrix_to_quaternion(rotation)

        return position, quaternion.tolist(), transform_robot

    def rotation_matrix_to_quaternion(self, rotation):
        trace = np.trace(rotation)

        # 数值稳定性处理
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # S=4*qw
            qw = 0.25 * s
            qx = (rotation[2, 1] - rotation[1, 2]) / s
            qy = (rotation[0, 2] - rotation[2, 0]) / s
            qz = (rotation[1, 0] - rotation[0, 1]) / s
        else:
            # 找出最大元素来避免数值问题
            i = np.argmax([rotation[0, 0], rotation[1, 1], rotation[2, 2]])
            if i == 0:
                s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
                qw = (rotation[2, 1] - rotation[1, 2]) / s
                qx = 0.25 * s
                qy = (rotation[0, 1] + rotation[1, 0]) / s
                qz = (rotation[0, 2] + rotation[2, 0]) / s
            elif i == 1:
                s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
                qw = (rotation[0, 2] - rotation[2, 0]) / s
                qx = (rotation[0, 1] + rotation[1, 0]) / s
                qy = 0.25 * s
                qz = (rotation[1, 2] + rotation[2, 1]) / s
            else:
                s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
                qw = (rotation[1, 0] - rotation[0, 1]) / s
                qx = (rotation[0, 2] + rotation[2, 0]) / s
                qy = (rotation[1, 2] + rotation[2, 1]) / s
                qz = 0.25 * s

        return np.array([qx, qy, qz, qw])

    def calculate_object_z_pose(self, qr_to_object_transform):
        # Transform from robot to camera, then camera to QR code, then QR code to object
        transform_robot_to_object = np.dot(self.cam2robot, qr_to_object_transform)
        position = transform_robot_to_object[:3, 3]

        # Assuming no rotation for Object Z
        orientation = [0.0, 0.0, 0.0, 1.0]
        return position, orientation

    def calculate_orientation_difference(self, target_orientation, current_orientation):
        # Quaternion difference calculation
        q1 = np.array(target_orientation)
        q2 = np.array(current_orientation)

        # Conjugate of target quaternion
        q1_conj = np.array([q1[0], -q1[1], -q1[2], -q1[3]])

        # Quaternion multiplication to get the difference
        q_diff = np.array([
            q1_conj[0] * q2[0] - q1_conj[1] * q2[1] - q1_conj[2] * q2[2] - q1_conj[3] * q2[3],
            q1_conj[0] * q2[1] + q1_conj[1] * q2[0] + q1_conj[2] * q2[3] - q1_conj[3] * q2[2],
            q1_conj[0] * q2[2] - q1_conj[1] * q2[3] + q1_conj[2] * q2[0] + q1_conj[3] * q2[1],
            q1_conj[0] * q2[3] + q1_conj[1] * q2[2] - q1_conj[2] * q2[1] + q1_conj[3] * q2[0]
        ])
        return q_diff.tolist()

    def quaternion_to_matrix(self, quaternion):
        """ 将四元数转换为 3x3 旋转矩阵 """
        q = quaternion / np.linalg.norm(quaternion)  # 归一化四元数
        q0, q1, q2, q3 = q
        R = np.array([[1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
                      [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
                      [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]])
        return R

    def position_to_matrix(self, position):
        """ 将位置向量转化为 4x4 的齐次变换矩阵 """
        T = np.eye(4)
        T[0:3, 3] = position
        return T

    def compute_transform(self, target_position, target_orientation, transform_qr_to_robot):
        """ 计算 target_qr_torobot 和 transform_current_to_target """

        # 将四元数转为旋转矩阵
        R_target = self.quaternion_to_matrix(target_orientation)

        # 目标变换矩阵
        target_qr_torobot = np.eye(4)
        target_qr_torobot[0:3, 0:3] = R_target
        target_qr_torobot[0:3, 3] = target_position

        # 计算当前到目标的变换矩阵
        # transform_current_to_target = np.dot(transform_qr_to_robot, np.linalg.inv(target_qr_torobot))
        # transform_current_to_target = np.dot(np.linalg.inv(target_qr_torobot), transform_qr_to_robot)
        transform_current_to_target = np.dot(np.linalg.inv(transform_qr_to_robot), target_qr_torobot)
        # transform_current_to_target = np.dot(target_qr_torobot, np.linalg.inv(transform_qr_to_robot))


        return target_qr_torobot, transform_current_to_target

    def save_target_pose(self):
        if self.target_pose is not None:
            target_data = {
                "position": self.target_pose[0].tolist(),
                "orientation": self.target_pose[1]
            }
            with open(self.config_file, 'w') as f:
                json.dump(target_data, f)
            self.get_logger().info("Target pose saved to configuration file.")

    def load_target_pose(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                target_data = json.load(f)
                self.target_pose = (
                    np.array(target_data["position"]),
                    target_data["orientation"]
                )
                self.get_logger().info("Target pose loaded from configuration file.")

    def spin_node(self):
        rclpy.spin(self)

    def stop(self):
        self.save_target_pose()
        self.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    rclpy.init()

    # 棋盘格点配置
    objp = np.zeros((2 * 2, 3), np.float32)
    grid = np.mgrid[0:2, 0:2].T.reshape(-1, 2)
    grid_sorted = np.zeros_like(grid)
    grid_sorted[:, 0] = grid[:, 0]
    grid_sorted[:, 1] = grid[:, 1]
    grid_sorted[2, :], grid_sorted[3, :] = grid[3, :], grid[2, :]

    # 将重排后的坐标赋值给 objp
    objp[:, :2] = grid_sorted
    worldpoint = objp * 115  # 棋盘格的宽度为50mm

    # Replace with your topic names
    rgb_topic = '/camera/color/image_raw'
    camera_info_topic = '/camera/color/camera_info'

    rgbd_camera_subscriber = RGBDCameraSubscriber(rgb_topic, camera_info_topic)

    # Define the QR code to Object transformation
    qr_to_object_transform = np.eye(4)
    qr_to_object_transform[:3, 3] = [50, 120, -479]  # x=50, y=120, z=479

    try:
        while rclpy.ok():
            if rgbd_camera_subscriber.mtx is not None and rgbd_camera_subscriber.rgb is not None:
                (corners, ids, rejected_corners) = stag.detectMarkers(rgbd_camera_subscriber.rgb, 17)

                if len(ids) >= 1:
                    for i, id in enumerate(ids):
                        success, rvec, tvec = cv2.solvePnP(worldpoint, corners[i],
                                                           rgbd_camera_subscriber.mtx,
                                                           rgbd_camera_subscriber.dist)
                        if success:
                            position, orientation , transform_qr_to_robot = rgbd_camera_subscriber.calculate_pose(rvec, tvec)

                            if rgbd_camera_subscriber.mode == 1 and rgbd_camera_subscriber.target_pose is None:
                                rgbd_camera_subscriber.target_pose = (position, orientation)
                                rgbd_camera_subscriber.get_logger().info("Target pose set.")

                            elif rgbd_camera_subscriber.mode == 2 and rgbd_camera_subscriber.target_pose is not None:
                                target_position, target_orientation = rgbd_camera_subscriber.target_pose
                                target_qr_torobot, transform_current_to_target = rgbd_camera_subscriber.compute_transform(target_position, target_orientation, transform_qr_to_robot)
                                object_position_navigation = transform_current_to_target[:3, 3]
                                object_orientation_navigation = rgbd_camera_subscriber.rotation_matrix_to_quaternion(transform_current_to_target[:3, :3])
                                rgbd_camera_subscriber.publish_pose(object_position_navigation, object_orientation_navigation)




                                # robotic arm
                                object_to_robot_transform = np.dot(transform_qr_to_robot, qr_to_object_transform)

                                # 提取物体的位置（平移部分）
                                object_position = object_to_robot_transform[:3, 3]
                                object_orientation = rgbd_camera_subscriber.rotation_matrix_to_quaternion(object_to_robot_transform[:3, :3])
                                rgbd_camera_subscriber.publish_object_pose(object_position, object_orientation)


    except KeyboardInterrupt:
        pass
    finally:
        rgbd_camera_subscriber.stop()