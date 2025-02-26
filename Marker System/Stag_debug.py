import stag
import cv2

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, CameraInfo
from cv_bridge import CvBridge

import threading
import time

import matplotlib.pyplot as plt



class RGBDCameraSubscriber(Node):
    def __init__(self, rgb_topic, camera_info_topic):
        super().__init__('rgbd_camera_subscriber')

        self.rgb_topic = rgb_topic
        self.camera_info_topic = camera_info_topic

        self.rgb = None  # Placeholder for the latest RGB frame
        self.mtx = None  # Camera intrinsic matrix
        self.dist = None  # Camera distortion coefficients

        self.bridge = CvBridge()

        # Data storage for accuracy calculation
        self.data = []  # Store (timestamp, x, y, z, roll, pitch, yaw)
        self.start_time = time.time()

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

        # Publisher for JointState
        self.joint_state_publisher = self.create_publisher(JointState, 'Marker_pose', 10)


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

    def publish_joint_state(self, id, position, rotation):
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        # Assign names as the detected marker IDs
        joint_state_msg.name = [str(id) for _ in range(6)]
        # Combine position and rotation into a flat list for all markers
        joint_state_msg.position = position + rotation
        self.joint_state_publisher.publish(joint_state_msg)

        # Record the data
        current_time = time.time()
        self.data.append((current_time, *position, *rotation))

        # Keep only data from the last 60 seconds
        self.data = [(t, x, y, z, roll, pitch, yaw) for t, x, y, z, roll, pitch, yaw in self.data if
                     current_time - t <= 60]

    def calculate_precision_error(self):
        if len(self.data) < 2:
            self.get_logger().info("Not enough data for precision calculation.")
            return None

        # Extract values for each dimension
        x_values = [entry[1] for entry in self.data]
        y_values = [entry[2] for entry in self.data]
        z_values = [entry[3] for entry in self.data]
        roll_values = [entry[4] for entry in self.data]
        pitch_values = [entry[5] for entry in self.data]
        yaw_values = [entry[6] for entry in self.data]

        # Calculate standard deviation for each dimension
        x_std = np.std(x_values)
        y_std = np.std(y_values)
        z_std = np.std(z_values)
        roll_std = np.std(roll_values)
        pitch_std = np.std(pitch_values)
        yaw_std = np.std(yaw_values)

        x_target = np.mean(np.array(x_values))
        y_target = np.mean(np.array(y_values))
        z_target = np.mean(np.array(z_values))
        roll_target = np.mean(np.array(roll_values))
        pitch_target = np.mean(np.array(pitch_values))
        yaw_target = np.mean(np.array(yaw_values))

        x_absolute_errors = np.abs(np.array(x_values) - x_target)
        y_absolute_errors = np.abs(np.array(y_values) - y_target)
        z_absolute_errors = np.abs(np.array(z_values) - z_target)
        roll_absolute_errors = np.abs(np.array(roll_values) - roll_target)
        pitch_absolute_errors = np.abs(np.array(pitch_values) - pitch_target)
        yaw_absolute_errors = np.abs(np.array(yaw_values) - yaw_target)

        # 平均误差
        x_mean_error = np.mean(x_absolute_errors)
        y_mean_error = np.mean(y_absolute_errors)
        z_mean_error = np.mean(z_absolute_errors)
        roll_mean_error = np.mean(roll_absolute_errors)
        pitch_mean_error = np.mean(pitch_absolute_errors)
        yaw_mean_error = np.mean(yaw_absolute_errors)

        # 极值误差
        x_max_error = np.max(x_absolute_errors)
        y_max_error = np.max(y_absolute_errors)
        z_max_error = np.max(z_absolute_errors)
        roll_max_error = np.max(roll_absolute_errors)
        pitch_max_error = np.max(pitch_absolute_errors)
        yaw_max_error = np.max(yaw_absolute_errors)



        precision_error = {
            "x_std": x_std,
            "y_std": y_std,
            "z_std": z_std,
            "roll_std": roll_std,
            "pitch_std": pitch_std,
            "yaw_std": yaw_std,
            "x_mean_error":x_mean_error,
            "y_mean_error":y_mean_error,
            "z_mean_error":z_mean_error,
            "roll_mean_error":roll_mean_error,
            "pitch_mean_error":pitch_mean_error,
            "yaw_mean_error":yaw_mean_error,
            "x_max_error":x_max_error,
            "y_max_error":y_max_error,
            "z_max_error":z_max_error,
            "roll_max_error:":roll_max_error,
            "pitch_max_error":pitch_max_error,
            "yaw_max_error":yaw_max_error,
        }

        self.get_logger().info(f"Precision error: {precision_error}")
        return precision_error

    def spin_node(self):
        rclpy.spin(self)

    def stop(self):
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
    worldpoint = objp * 77.6  # 棋盘格的宽度为50mm

    # Replace with your topic names
    rgb_topic = '/camera/color/image_raw'
    camera_info_topic = '/camera/color/camera_info'

    rgbd_camera_subscriber = RGBDCameraSubscriber(rgb_topic, camera_info_topic)

    try:
        while rclpy.ok():
            if rgbd_camera_subscriber.mtx is not None and rgbd_camera_subscriber.rgb is not None:
                gray_image = cv2.cvtColor(rgbd_camera_subscriber.rgb, cv2.COLOR_BGR2GRAY)
                (corners, ids, rejected_corners) = stag.detectMarkers(rgbd_camera_subscriber.rgb, 17)
                if len(ids) >= 1:
                    for i, id in enumerate(ids):
                        # 亚像素精度的角点优化
                        corners_i = corners[i]
                        corners_i = np.array(corners_i, dtype=np.float32)  # 确保角点数据类型正确

                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
                        cv2.cornerSubPix(gray_image, corners_i, (5, 5), (-1, -1), criteria)  # 优化角点
                        # 提取并绘制原始图像和优化后的图像
                        img_copy = rgbd_camera_subscriber.rgb.copy()
                        optimized_img = rgbd_camera_subscriber.rgb.copy()

                        # 放大角点区域的尺寸
                        x_min, x_max = int(np.min(corners_i[:, 0]) - 10), int(np.max(corners_i[:, 0]) + 10)
                        y_min, y_max = int(np.min(corners_i[:, 1]) - 10), int(np.max(corners_i[:, 1]) + 10)

                        # 提取放大的区域
                        zoomed_in_img_copy = img_copy[y_min:y_max, x_min:x_max]
                        zoomed_in_optimized_img = optimized_img[y_min:y_max, x_min:x_max]

                        # 绘制原始角点
                        for corner in corners[i][0]:
                            print("corners[i]:", corners[i][0])
                            x, y = corner.ravel()
                            cv2.circle(img_copy, (int(x), int(y)), 3, (0, 0, 255), -1)  # 红色圆圈

                        # 绘制优化后的角点
                        for corner in corners_i[0]:
                            print("corners_i:", corners_i)

                            x, y = corner.ravel()
                            cv2.circle(optimized_img, (int(x), int(y)), 3, (0, 255, 0), -1)  # 绿色圆圈

                        # 将左图和右图放在一起
                        combined_img = np.hstack([zoomed_in_img_copy, zoomed_in_optimized_img])

                        # 显示对比图
                        plt.figure(figsize=(12, 6))
                        plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
                        plt.title('Left: Original Corners, Right: Optimized Corners')
                        plt.axis('off')
                        plt.show()


                        # success, rvec, tvec = cv2.solvePnP(worldpoint, corners_i, rgbd_camera_subscriber.mtx, rgbd_camera_subscriber.dist)
                        success, rvec, tvec = cv2.solvePnP(worldpoint, corners[i], rgbd_camera_subscriber.mtx, rgbd_camera_subscriber.dist)
                        if success:
                            rotation_matrix, _ = cv2.Rodrigues(rvec)
                            position = tvec.flatten()
                            # Convert rotation matrix to Euler angles (roll, pitch, yaw)
                            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                            pitch = np.arctan2(-rotation_matrix[2, 0],np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))
                            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                            rgbd_camera_subscriber.publish_joint_state(id, position.tolist(), [roll, pitch, yaw])

            # Calculate precision error every 10 seconds
            if int(time.time()) % 60 == 0:
                rgbd_camera_subscriber.calculate_precision_error()

    except KeyboardInterrupt:
        pass
    finally:
        rgbd_camera_subscriber.stop()