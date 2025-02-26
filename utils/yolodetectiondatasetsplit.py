import os
import shutil
import random

# 设置文件夹路径
base_folder = '/home/lz/Robotic_vision_kepler/baoma20250212'
source_folder = '/home/lz/Robotic_vision_kepler/bao'  # 替换为你的源文件夹路径

# 创建目标文件夹结构
train_images_folder = os.path.join(base_folder, 'train', 'images')
train_labels_folder = os.path.join(base_folder, 'train', 'labels')
val_images_folder = os.path.join(base_folder, 'val', 'images')
val_labels_folder = os.path.join(base_folder, 'val', 'labels')

os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

# 获取所有jpg和txt文件
jpg_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]
txt_files = [f for f in os.listdir(source_folder) if f.endswith('.txt')]

# 找出可以配对的jpg和txt文件
paired_files = []
for jpg_file in jpg_files:
    txt_file = jpg_file.replace('.jpg', '.txt')
    if txt_file in txt_files:
        paired_files.append((jpg_file, txt_file))

# 随机打乱配对文件列表
random.shuffle(paired_files)

# 按照8:2的比例分割文件
split_index = int(len(paired_files) * 0.8)
train_files = paired_files[:split_index]
val_files = paired_files[split_index:]

# 将文件复制到相应的文件夹
for jpg_file, txt_file in train_files:
    shutil.copy(os.path.join(source_folder, jpg_file), os.path.join(train_images_folder, jpg_file))
    shutil.copy(os.path.join(source_folder, txt_file), os.path.join(train_labels_folder, txt_file))

for jpg_file, txt_file in val_files:
    shutil.copy(os.path.join(source_folder, jpg_file), os.path.join(val_images_folder, jpg_file))
    shutil.copy(os.path.join(source_folder, txt_file), os.path.join(val_labels_folder, txt_file))

print("文件已成功分配到train和val文件夹中。")