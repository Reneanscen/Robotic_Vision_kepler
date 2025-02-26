import os
from PIL import Image
import numpy as np

def CLIP(value, down, top):
    if value <= down:
        return down
    if value >= top:
        return top
    return value

def check_roi_region(roi_region, w, h):
    safe_roi_region = [0, w-2, 0, h-2]
    if (roi_region[0] == 0 and roi_region[1] == 0 and roi_region[2] == 0 and roi_region[3] == 0):
        return safe_roi_region
    if (roi_region[1] < roi_region[0]) or (roi_region[3] < roi_region[2]):
        print("check your roi_region!")
        return safe_roi_region

    roi_region[0] = CLIP(roi_region[0], 0, w-2)
    roi_region[1] = CLIP(roi_region[1], 0, w - 2)
    roi_region[2] = CLIP(roi_region[2], 0, h - 2)
    roi_region[3] = CLIP(roi_region[3], 0, h - 2)
    return roi_region

def cal_resolution(img_gray, roi_region):
    resolution = 0
    w, h = img_gray.shape
    start_x, end_x, start_y, end_y = check_roi_region(roi_region, w, h)
    dx = np.sum(np.abs(np.gradient(img_gray[start_x:end_x, start_y:end_y], axis=1)))  # 水平方向梯度
    dy = np.sum(np.gradient(img_gray[start_x:end_x, start_y:end_y], axis=0))  # 垂直方向梯度
    resolution = dx + dy
    return resolution

def main():
    imgs_dir = r'/home/lz/Robotic_vision_kepler/images'
    roi_region = [0, 0, 0, 0]    #可设置计算清晰度的矩形区域，如只计算人脸区域，默认全图计算

    imgs_names = os.listdir(imgs_dir)
    imgs_names = [file for file in imgs_names if file.lower().endswith('.jpg')]
    print("图像数量：", len(imgs_names))

    count = 1
    img_resolution_list = []
    for img in imgs_names:
        img_path = os.path.join(imgs_dir, img)
        img_src = Image.open(img_path)
        img_gray = np.array(img_src.convert('L'))
        img_resolution = cal_resolution(img_gray, roi_region)
        img_resolution_list.append(img_resolution)
        print(count, img_path, img_resolution)
        count += 1

    imgs_names_and_resolution_list = list(zip(imgs_names, img_resolution_list))
    sorted_resolution_list = sorted(imgs_names_and_resolution_list, key=lambda x: x[1], reverse=True)
    print("=========以下按清晰度从高到低排序=========")
    for idx, img in enumerate(sorted_resolution_list):
        print(idx + 1, img)


if __name__ == "__main__":
    main()

