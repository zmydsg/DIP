import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from skimage import io, color
from scipy.spatial.distance import cdist


# 读取图像并转换为灰度图像
image = io.imread('F:/DIP/experiment6/origin.png')
image_rgb = image[..., :3]  # 只取前3个通道（RGB）
gray_image = color.rgb2gray(image_rgb)

# 将图像转换为一个像素的列表
height, width = gray_image.shape
pixels = np.array([gray_image[i, j] for i in range(height) for j in range(width)])

# 构建图的相似度矩阵（使用高斯核函数）
sigma = 0.1  # 高斯相似度的参数
W = np.exp(-cdist(pixels, pixels, metric='sqeuclidean') / (2 * sigma ** 2))

# 计算度矩阵 D
D = np.diag(W.sum(axis=1))

# 计算拉普拉斯矩阵 L = D - W
L = D - W

# 计算拉普拉斯矩阵的特征向量
eigvals, eigvecs = np.linalg.eigh(L)

# 选择最小的特征向量
eigenvector = eigvecs[:, 1]

# 使用谱聚类进行分割
sc = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='kmeans')
labels = sc.fit_predict(W)

# 显示结果
segmented_image = labels.reshape(height, width)
plt.imshow(segmented_image, cmap='gray')
plt.title('Normalized Cut Segmentation')
plt.show()
