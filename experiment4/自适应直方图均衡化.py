import cv2
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('F:\DIP\experiment4\originalimages1.png', cv2.IMREAD_GRAYSCALE)

# 创建CLAHE对象，并对图像进行自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
equalized_image = clahe.apply(image)

# 使用高斯滤波器降噪
blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 1.4)

# 设置Canny算子的阈值
lower_threshold = 50
upper_threshold = 150

# 应用Canny边缘检测
edges = cv2.Canny(blurred_image, lower_threshold, upper_threshold)

cv2.imwrite("F:\DIP\experiment4\Result1.png",edges)
# 显示原始图像和边缘检测结果
#plt.figure(figsize=(5, 5))
# plt.subplot(1, 3, 1)
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')

# plt.subplot(1, 3, 2)
# plt.title('CLAHE Enhanced Image')
# plt.imshow(equalized_image, cmap='gray')

# plt.subplot(1, 3, 3)
# plt.title('Canny Edges')
# plt.imshow(edges, cmap='gray')

# plt.tight_layout()
# plt.show()
