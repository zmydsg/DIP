import json
import numpy as np
import cv2
import os

# 加载VIA导出的标注文件
with open('F:/DIP/via111.json') as f:
    annotations = json.load(f)

image_dir = 'F:/DIP/origial images/'  # 原图路径
mask_dir = 'F:/DIP/'  # 掩码保存路径

# 创建保存掩码的目录（如果不存在）
os.makedirs(mask_dir, exist_ok=True)

# 遍历每个图像的标注数据
for image_key, image_data in annotations.items():
    image_name = image_data['filename']  # 获取图像名称
    
    # 加载原始图像以获取尺寸
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Image {image_name} not found, skipping...")
        continue
    
    height, width, _ = image.shape  # 获取图像尺寸
    
    # 创建一个空白的掩码图像（全黑）
    mask = np.zeros((height, width), dtype=np.uint8)

    # 获取标注的区域（regions）
    regions = image_data['regions']
    
    for region in regions:
        shape_attributes = region['shape_attributes']
        points_x = shape_attributes['all_points_x']
        points_y = shape_attributes['all_points_y']

        # 将折线（polyline）转化为多边形（polygon）
        polygon = np.array([[(x, y) for x, y in zip(points_x, points_y)]], dtype=np.int32)
        
        # 使用OpenCV将多边形填充为白色（像素值255）
        cv2.fillPoly(mask, polygon, 255)

    # 保存生成的掩码图像
    mask_path = os.path.join(mask_dir, os.path.splitext(image_name)[0] + '_mask.png')
    cv2.imwrite(mask_path, mask)

    print(f'Saved mask for {image_name} at {mask_path}')
