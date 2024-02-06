import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as  np
# 定义标签文件夹和图像文件夹的路径
labels_folder = "/home/ubuntu/DataSet/FULL_middle/test/labels"
images_folder = "/home/ubuntu/GITHUG/ultralytics/runs/detect/predict5"
output_folder = os.path.join(images_folder,"labeled")
os.mkdir(output_folder)
# 获取标签文件列表
label_files = os.listdir(labels_folder)

# 循环处理每个标签文件
for label_file in label_files:
    # 构建对应的图像文件路径
    try:
        image_file = os.path.join(images_folder, os.path.splitext(label_file)[0] + ".tif")

        # 读取图像
        # image = Image.open(image_file)
        image = cv2.imread(image_file)
        if image.all()==None:
            raise IOError
    except:
        image_file = os.path.join(images_folder, os.path.splitext(label_file)[0] + ".png")

        # 读取图像
        # image = Image.open(image_file)
        image = cv2.imread(image_file)

    image_array = np.array(image)

    # 读取标签文件
    with open(os.path.join(labels_folder, label_file), 'r') as f:
        lines = f.readlines()

    # 循环处理每个检测框
    for line in lines:
        # 解析检测框坐标信息
        box_info = [float(coord) for coord in line.strip().split()]
        x, y, w, h = box_info[1:5]

        # 计算检测框的坐标在图像中的位置
        x1 = int((x - w / 2) * image_array.shape[1])
        y1 = int((y - h / 2) * image_array.shape[0])
        x2 = int((x + w / 2) * image_array.shape[1])
        y2 = int((y + h / 2) * image_array.shape[0])

        # 在图像上画红色框
        cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 设置颜色为红色
        # cv2.putText(image_array, f"{confidence:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
        #             cv2.LINE_AA)
    # 转换图像为灰度
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGBA)

    output_file = os.path.join(output_folder, os.path.splitext(label_file)[0] + "_output.png")
    cv2.imwrite(output_file, cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
