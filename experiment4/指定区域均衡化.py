import cv2
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('F:\DIP\experiment4\originalimages1.png', cv2.IMREAD_GRAYSCALE)

# 定义感兴趣区域 (ROI) 的位置和大小 (y1:y2, x1:x2)
roi_y1, roi_y2 = 100, 200  # y方向的起点和终点
roi_x1, roi_x2 = 150, 250  # x方向的起点和终点

# 提取感兴趣区域
roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

# 创建CLAHE对象，对感兴趣区域进行自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized_roi = clahe.apply(roi)

# 将处理后的ROI替换回原始图像
image_with_equalized_roi = image.copy()  # 复制原始图像，保留原图
image_with_equalized_roi[roi_y1:roi_y2, roi_x1:roi_x2] = equalized_roi

# # 显示原始图像、处理后的ROI和结果图像
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')

# plt.subplot(1, 3, 2)
# plt.title('Equalized ROI')
# plt.imshow(equalized_roi, cmap='gray')

# plt.subplot(1, 3, 3)
# plt.title('Image with Equalized ROI')
# plt.imshow(image_with_equalized_roi, cmap='gray')

# plt.tight_layout()
# plt.show()

# 使用高斯滤波器降噪
blurred_image = cv2.GaussianBlur(image_with_equalized_roi, (5, 5), 1.4)

# 设置Canny算子的阈值
lower_threshold = 50
upper_threshold = 150

# 应用Canny边缘检测
edges = cv2.Canny(blurred_image, lower_threshold, upper_threshold)

# 显示原始图像和边缘检测结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('CLAHE Enhanced Image')
plt.imshow(image_with_equalized_roi, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Canny Edges')
plt.imshow(edges, cmap='gray')

plt.tight_layout()
plt.show()