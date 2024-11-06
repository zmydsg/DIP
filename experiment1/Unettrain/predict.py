# predict.py

import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# 加载训练好的模型
model = load_model('unet_yellow_segmentation.h5')

# 读取要进行推断的图像
image = cv2.imread('data/images/exp1.png')
image_resized = cv2.resize(image, (256, 256))  # 调整图像尺寸
image_norm = image_resized / 255.0  # 归一化
image_input = np.expand_dims(image_norm, axis=0)  # 增加一个维度

# 进行预测
predicted_mask = model.predict(image_input)

# 将预测的结果阈值化，生成二值掩码
predicted_mask = (predicted_mask[0, :, :, 0] > 0.5).astype(np.uint8)

# 显示原图像和预测的掩码
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(predicted_mask, cmap='gray')
plt.title("Predicted Mask")

plt.show()
