import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io, color
from scipy.spatial.distance import cdist

# 读取图像并确保它是RGB图像（如果是RGBA格式，则去掉Alpha通道）
image = io.imread('F:/DIP/experiment6/origin.png')

# 如果图像是RGBA格式，去掉Alpha通道
if image.shape[2] == 4:
    image_rgb = image[:, :, :3]  # 只保留RGB通道
else:
    image_rgb = image

# 转换为Lab色彩空间，Lab对颜色差异更加敏感
image_rgb = color.rgb2lab(image_rgb)

# 获取图像的尺寸
height, width, _ = image_rgb.shape
pixels = np.reshape(image_rgb, (-1, 3))  # 将图像展平，得到每个像素的颜色特征

# 只计算相邻像素之间的相似度
W = np.zeros((pixels.shape[0], pixels.shape[0]))

# 遍历每个像素，计算与相邻像素的相似度
for i in range(height):
    for j in range(width):
        idx = i * width + j  # 当前像素的索引
        if i < height - 1:  # 下方像素
            down_idx = (i + 1) * width + j
            W[idx, down_idx] = np.exp(-np.sum((pixels[idx] - pixels[down_idx]) ** 2) / (2 * 0.1 ** 2))
        if j < width - 1:  # 右侧像素
            right_idx = i * width + (j + 1)
            W[idx, right_idx] = np.exp(-np.sum((pixels[idx] - pixels[right_idx]) ** 2) / (2 * 0.1 ** 2))

# 计算度矩阵D
D = np.diag(W.sum(axis=1))

# 计算拉普拉斯矩阵L
L = D - W

# 计算L的特征值和特征向量
eigvals, eigvecs = np.linalg.eigh(L)

# 选择最小的k个特征向量进行聚类
k = 2  # 聚成2类，即前景和背景
feature_vectors = eigvecs[:, 1:k+1]  # 选择第二个到第k+1个特征向量

# 使用k-means聚类
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(feature_vectors)

# 将标签恢复为图像形状
segmented_image = np.reshape(labels, (height, width))

# 可视化结果
plt.imshow(segmented_image, cmap='jet')
plt.title('Normalized Cut Segmentation')
plt.show()
