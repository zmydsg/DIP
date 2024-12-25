import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

# 读取图像
image_path = "F:/DIP/final_project/original3.png"  # 替换为实际的图像路径
image = img_as_float(io.imread(image_path))

# 只保留前三个通道（RGB）
if image.shape[-1] == 4:
    image = image[:, :, :3]

# 应用SLIC算法
# n_segments: 超像素数目
# compactness: 紧凑性参数，数值越大，超像素的形状越规则
segments = slic(image, n_segments=15, compactness=10, start_label=1)

# 可视化结果
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mark_boundaries(image, segments))
plt.title("SLIC Segmentation")
plt.axis("off")

plt.tight_layout()
plt.show()
