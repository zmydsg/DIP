import cv2
import numpy as np

def canny_edge_detection(image_path, low_threshold=100, high_threshold=200):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 使用高斯滤波器去噪
    blurred_img = cv2.GaussianBlur(img, (5, 5), 1.4)

    # 进行Canny边缘检测
    edges = cv2.Canny(blurred_img, low_threshold, high_threshold)

    return edges

# 示例使用
image_path = 'F:\DIP\experiment4\originalimages1.png'  # 替换为你的图像路径
edges = canny_edge_detection(image_path)

cv2.imwrite("F:\DIP\experiment4\Result2.png",edges)
