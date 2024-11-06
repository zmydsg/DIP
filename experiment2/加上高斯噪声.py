import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
image = cv2.imread('F:/DIP/experiment2/original1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 获取图片的形状
row, col, ch = image.shape

# 生成严重的高斯噪声
mean = 0
var = 5000  # 方差
sigma = var ** 0.5
gaussian_noise = np.random.normal(mean, sigma, (row, col, ch))


# 将噪声加到图片上
noisy_image = np.clip(image + gaussian_noise, 0, 255).astype(np.uint8)


# 应用高斯滤波进行去噪
filtered_image = cv2.GaussianBlur(noisy_image, (7, 7), 0)  # (5, 5) 是滤波器的大小

# 显示原始、加噪声以及滤波后的图片
#plt.figure(figsize=(5,10))

# 原始图像
# plt.subplot(3, 1, 1)
# plt.title('Original Image')
# plt.imshow(image)
# plt.axis('off')

# 带有高斯噪声的图像
# plt.subplot(2, 1, 1)
# plt.title('Noisy Image with Severe Gaussian Noise')
# plt.imshow(noisy_image)
# plt.axis('off')

# 滤波后的图像
# plt.subplot(2, 2, 1)
# plt.title('Filtered Image (Gaussian Blur)')
# plt.imshow(filtered_image)
# plt.axis('off')

# plt.tight_layout()
# plt.show()

# 保存滤波后的图片
cv2.imwrite('F:/DIP/experiment2/noisy_image2.png', cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))
cv2.imwrite('F:/DIP/experiment2/filtered_image2.png', cv2.cvtColor(filtered_image, cv2.COLOR_RGB2BGR))
