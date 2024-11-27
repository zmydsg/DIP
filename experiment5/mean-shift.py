import cv2
import numpy as np

# # 读取图像
# image = cv2.imread('F:/DIP/experiment5/origin.png')
# # 转换为Lab颜色空间
# image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# # 使用pyrMeanShiftFiltering进行初步分割
spatial_radius = 10
color_radius = 20
max_pyramid_level = 1
# segmented_image = cv2.pyrMeanShiftFiltering(image_lab, spatial_radius, color_radius, max_pyramid_level)

# # 转换为灰度图像
# segmented_gray = cv2.cvtColor(segmented_image, cv2.COLOR_Lab2GRAY)

# # 使用Canny边缘检测
# edges = cv2.Canny(segmented_gray, 50, 150)

# # 将边缘绘制在原始图像上
# edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# enhanced_image = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)

# # 显示增强的分割结果
# cv2.imshow('Enhanced Segmentation with Canny Edges', enhanced_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 读取图像
image = cv2.imread('F:/DIP/experiment5/origin.png')
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# 使用均值漂移进行初步分割
segmented_image = cv2.pyrMeanShiftFiltering(image_lab, spatial_radius, color_radius, max_pyramid_level)

# 将BGR分割结果转换为灰度图像
segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_Lab2BGR)
segmented_gray = cv2.cvtColor(segmented_image_bgr, cv2.COLOR_BGR2GRAY)

# 继续你的边缘检测或者其他操作
edges = cv2.Canny(segmented_gray, 50, 150)

# 将边缘绘制在原始图像上
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
enhanced_image = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)

# 显示结果
cv2.imshow('Enhanced Segmentation with Canny Edges', enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
