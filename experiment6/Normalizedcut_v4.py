



import cv2
import numpy as np

# 1. 读取图像并转换为灰度图
img = cv2.imread("F:/DIP/experiment6/origin.png", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 自适应平滑 - 使用双边滤波（bilateralFilter）
# 参数说明：
# d: 邻域直径，可根据图像大小调整，如9或更大
# sigmaColor: 色彩空间滤波器sigma值，数字越大，越模糊颜色相近的区域
# sigmaSpace: 坐标空间滤波器sigma值，数字越大，越大范围的像素会被平滑考虑
smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=25, sigmaSpace=15)

# 3. 创建CLAHE对象并应用
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
clahe_img = clahe.apply(smoothed)

# 4. 显示与保存结果
cv2.imshow('Original Gray', gray)
cv2.imshow('Smoothed', smoothed)
cv2.imshow('CLAHE Result', clahe_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite('clahe_result.jpg', clahe_img)
