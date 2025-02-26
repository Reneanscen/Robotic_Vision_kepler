import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import stag  # Assuming stag is the module you're using for marker detection

class PoseEstimator(Node):
    def __init__(self):
        super().__init__('pose_estimator')

        # Topics
        self.rgb_topic = '/camera/color/image_raw'
        self.camera_info_topic = '/camera/color/camera_info'
        self.pose_topic = '/camera/pose'

        # Subscribers
        self.image_subscriber = self.create_subscription(Image, self.rgb_topic, self.image_callback, 10)
        self.camera_info_subscriber = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        # Publisher
        self.pose_publisher = self.create_publisher(Pose, self.pose_topic, 10)

        # Data
        self.bridge = CvBridge()
        self.rgb = None
        self.mtx = None
        self.dist = None

        # Prepare object points for a 2x2 grid (2x2 chessboard for detection)
        objp = np.zeros((2 * 2, 3), np.float32)
        grid = np.mgrid[0:2, 0:2].T.reshape(-1, 2)
        grid_sorted = np.zeros_like(grid)
        grid_sorted[:, 0] = grid[:, 0]
        grid_sorted[:, 1] = grid[:, 1]
        grid_sorted[2, :], grid_sorted[3, :] = grid[3, :], grid[2, :]
        objp[:, :2] = grid_sorted
        self.worldpoint = objp * 115  # Each grid square is 115mm

        # Mode flag: 0 for Mode 1, 1 for Mode 2
        self.mode = 0
        self.target_rvec = None  # Target's rotation vector in Mode 1
        self.target_tvec = None  # Target's translation vector in Mode 1

    def image_callback(self, msg):
        try:
            self.rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def camera_info_callback(self, msg):
        if self.mtx is None and self.dist is None:
            self.mtx = np.array(msg.k).reshape(3, 3)
            self.dist = np.array(msg.d)

    def publish_pose(self, rvec, tvec):
        pose = Pose()
        pose.position.x = tvec[0][0]
        pose.position.y = tvec[1][0]
        pose.position.z = tvec[2][0]

        # Convert rvec to quaternion
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        q = self.rotation_matrix_to_quaternion(rotation_matrix)

        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        self.pose_publisher.publish(pose)

    @staticmethod
    def rotation_matrix_to_quaternion(rot_matrix):
        tr = np.trace(rot_matrix)
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * s
            qx = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
            qy = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
            qz = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
        elif (rot_matrix[0, 0] > rot_matrix[1, 1]) and (rot_matrix[0, 0] > rot_matrix[2, 2]):
            s = np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2]) * 2
            qw = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            qz = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
        elif rot_matrix[1, 1] > rot_matrix[2, 2]:
            s = np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2]) * 2
            qw = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
            qx = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1]) * 2
            qw = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
            qx = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
            qy = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
            qz = 0.25 * s
        return [qx, qy, qz, qw]

    def process(self):
        if self.rgb is not None and self.mtx is not None:
            gray = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            corners, ids, rejected_corners = stag.detectMarkers(self.rgb, 17)  # Assuming marker ID 17 is used

            if len(ids) >= 1:
                for i, id in enumerate(ids):
                    # 将 corners 转换为 numpy 数组，确保可以进行修改
                    corners = np.array(corners)

                    # 亚像素精度优化
                    corners[i] = cv2.cornerSubPix(gray, corners[i], (5, 5), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))

                    # Mode 1: Record the relative pose of the target marker to the camera
                    if self.mode == 0:
                        success, rvec, tvec = cv2.solvePnP(self.worldpoint, corners[i], self.mtx, self.dist)
                        if success:
                            self.target_rvec = rvec
                            self.target_tvec = tvec
                            self.get_logger().info(f"Target pose recorded: rvec={rvec}, tvec={tvec}")
                            self.mode = 1  # Switch to Mode 2
                    # Mode 2: Compute the transformation from the current position to the target position
                    elif self.mode == 1:
                        success, rvec, tvec = cv2.solvePnP(self.worldpoint, corners[i], self.mtx, self.dist)
                        if success:
                            # Compute the relative transformation from the current pose to the target pose
                            R_cam_to_marker, _ = cv2.Rodrigues(rvec)
                            R_target_to_marker, _ = cv2.Rodrigues(self.target_rvec)
                            t_cam_to_marker = tvec
                            t_target_to_marker = self.target_tvec

                            # Compute the transformation from the current position to the target position
                            R_target_to_cam = R_target_to_marker.T
                            t_target_to_cam = -R_target_to_cam @ t_target_to_marker

                            # Calculate the final pose transformation (in world coordinates)
                            t_world = R_target_to_cam @ t_cam_to_marker + t_target_to_cam

                            # Publish the calculated pose
                            self.publish_pose(rvec, t_world)

def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimator()

    try:
        while rclpy.ok():
            rclpy.spin_once(node)
            node.process()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
