import cv2
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('F:\DIP\experiment5\K-means_result.png', cv2.IMREAD_GRAYSCALE)

# 使用高斯滤波器降噪
blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

# 设置Canny算子的阈值
lower_threshold = 38
upper_threshold = 120

# 应用Canny边缘检测
edges = cv2.Canny(blurred_image, lower_threshold, upper_threshold)

# # 显示原始图像和边缘检测结果
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')

# plt.subplot(1, 2, 2)
# plt.title('Canny Edges')
# plt.imshow(edges, cmap='gray')

# plt.show()

cv2.imwrite('F:\DIP\DIP_finalpaper/canny_result.png',edges)