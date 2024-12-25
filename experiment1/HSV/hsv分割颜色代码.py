import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
image = cv2.imread('F:\DIP\experiment2\original3.png')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义黄色的 HSV 范围
lower_blue = np.array([100, 150, 0])   # 蓝色的下限
upper_blue = np.array([140, 255, 255])   # 蓝色的上限

# 应用阈值分割，生成掩码
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# 使用掩码提取黄色区域
yellow_extracted = cv2.bitwise_and(image, image, mask=mask)

# 显示原始图片和提取的黄色区域
plt.figure(figsize=(10, 5))

# 显示原始图片
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# 显示提取的黄色区域
plt.subplot(1, 2, 2)
plt.title('Yellow Extracted by Thresholding')
plt.imshow(cv2.cvtColor(yellow_extracted, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

# 保存提取的黄色区域
cv2.imwrite('F:/DIP/experiment1/HSV/yellow_extracted_threshold3.png', yellow_extracted)
