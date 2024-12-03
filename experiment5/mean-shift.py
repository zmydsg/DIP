import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
image_path = 'F:/DIP/experiment5/origin.png'  # 替换为你的图片路径
image = cv2.imread(image_path)

# 检查是否成功加载图像
if image is None:
    print("Error loading image!")
    exit()

# 将图像从BGR转换为RGB（OpenCV默认为BGR）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 自适应直方图均衡（CLAHE）
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 创建CLAHE对象
equalized_image = clahe.apply(gray_image)  # 应用自适应直方图均衡

# 显示原始图像和均衡后的图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('CLAHE Image')
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.show()

# 使用MeanShift进行前景提取

# 转换为二维数据点进行MeanShift处理
# 将图像转为二维数据（每个像素是一个点，RGB颜色值作为特征）
reshaped_image = image.reshape((-1, 3))  # 每个像素作为一个数据点
reshaped_image = np.float32(reshaped_image)  # 转换为浮点数

# 设置MeanShift的带宽
bandwidth = 30  # 可以根据需要调整此参数

# 使用OpenCV的MeanShift实现
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
_, labels, centers = cv2.kmeans(reshaped_image, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 将聚类结果恢复到原图像的尺寸
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)

# 转换回RGB格式以显示
segmented_image_rgb = cv2.cvtColor(np.uint8(segmented_image), cv2.COLOR_BGR2RGB)

# 显示前景提取结果
plt.figure(figsize=(8, 8))
plt.imshow(segmented_image_rgb)
plt.title('Tiger Foreground Extraction using Mean Shift')
plt.axis('off')
plt.show()


