import sys
import numpy as np
from skimage import io
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
import cv2



# 加载图像
img = cv2.imread("F:/DIP/final_project/original3.png", cv2.IMREAD_COLOR)
num_segments = 200
compactness = 4

# 获取分割数和紧凑度
num_segments = int(sys.argv[2])
compactness = int(sys.argv[3])

# 使用 SLIC 算法进行图像分割
segments = slic(img, n_segments=num_segments, compactness=compactness, start_label=1)

# 标记分割的边界
boundaries = mark_boundaries(img, segments)

# 展示结果图像
plt.imshow(boundaries)
plt.axis('off')  # 去除坐标轴
plt.show()

# # 保存结果为图片（转换为 uint8 类型后保存）
# io.imsave("SLICimg.jpg", (boundaries * 255).astype(np.uint8))
