import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.bridge = CvBridge()
        self.video_writer = None
        self.subscriber = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.get_logger().info("Subscribed to /camera/color/image_raw")

    def image_callback(self, msg):
        try:
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            if self.video_writer is None:
                # 获取图像的尺寸
                height, width, _ = cv_image.shape
                # 创建VideoWriter对象，设置视频参数
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4编码
                self.video_writer = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (width, height))
                self.get_logger().info("Video recording started")

            # 写入视频帧
            self.video_writer.write(cv_image)

        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")

    def destroy(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info("Video recording stopped")

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraSubscriber()

    try:
        rclpy.spin(camera_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        camera_subscriber.destroy()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
