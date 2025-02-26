import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import os
import datetime

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.image_count = 0  # 用于编号
        self.save_dir = "images"  # 指定保存目录
        os.makedirs(self.save_dir, exist_ok=True)  # 确保目录存在

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{self.image_count}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, cv_image)
        self.get_logger().info(f"Image saved: {filepath}")
        self.image_count += 1  # 更新编号
        if self.image_count > 100:
            self.image_count = 0


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()