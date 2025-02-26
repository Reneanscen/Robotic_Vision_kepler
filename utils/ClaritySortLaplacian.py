import cv2
import numpy as np
import os
import shutil

def calculate_sharpness(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算拉普拉斯变换（图像的二阶导数）
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # 计算拉普拉斯变换的方差
    variance = laplacian.var()

    return variance


def main():
    imgs_dir = r'/home/lz/Robotic_vision_kepler/ObjectDection/data/images'  # 设置你的图像文件夹路径
    target_dir = r'/home/lz/Robotic_vision_kepler/baoma'  # 设置你要复制到的目标文件夹路径

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)  # 如果目标文件夹不存在，创建它

    imgs_names = os.listdir(imgs_dir)
    imgs_names = [file for file in imgs_names if file.lower().endswith(('.jpg', '.jpeg', '.png'))]  # 只处理图片文件
    print("图像数量：", len(imgs_names))

    # 用于存储每个图像的清晰度值
    img_sharpness_list = []

    # 遍历文件夹中的每张图像，计算清晰度
    for img_name in imgs_names:
        img_path = os.path.join(imgs_dir, img_name)

        # 读取图像
        img = cv2.imread(img_path)

        if img is None:
            print(f"无法读取图像: {img_name}")
            continue

        # 计算该图像的清晰度（拉普拉斯方差）
        sharpness = calculate_sharpness(img)

        # 将图像名称和清晰度值存储在列表中
        img_sharpness_list.append((img_name, sharpness))
        print(f"{img_name} 清晰度: {sharpness}")

        # 如果清晰度小于 820，删除该图像
        if sharpness < 820:
            print(f"删除图像: {img_name} (清晰度小于820)")
            os.remove(img_path)

    # 按照清晰度进行排序，从高到低
    sorted_sharpness_list = sorted(img_sharpness_list, key=lambda x: x[1], reverse=True)

    # 打印按清晰度排序的图像列表
    print("========= 以下按清晰度从高到低排序 =========")
    for idx, (img_name, sharpness) in enumerate(sorted_sharpness_list):
        print(f"{idx + 1}. {img_name} - 清晰度: {sharpness}")

    # # 复制前 3000 张清晰度最高的图像到目标文件夹
    # count = 0
    # for idx, (img_name, sharpness) in enumerate(sorted_sharpness_list[:3000]):
    #     img_path = os.path.join(imgs_dir, img_name)
    #     target_img_path = os.path.join(target_dir, img_name)
    #
    #     # 复制文件
    #     shutil.copy(img_path, target_img_path)
    #     print(f"复制 {img_name} 到 {target_dir}")
    #     count += 1
    #
    # print(f"已成功复制 {count} 张图像到目标文件夹。")


if __name__ == "__main__":
    main()
