import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.spatial.distance import cdist
import networkx as nx

def min_cut_segmentation(image, sigma=0.1, k=2):
    # 将图像转换为Lab色彩空间（因为Lab颜色空间对颜色差异更敏感）
    image_rgb = color.rgb2lab(image)
    height, width, _ = image_rgb.shape
    pixels = np.reshape(image_rgb, (-1, 3))  # 将图像展平为像素的RGB特征

    # 构建相似度矩阵W（计算每对像素之间的相似度）
    distances = cdist(pixels, pixels, 'sqeuclidean')  # 计算每对像素的欧几里得距离
    W = np.exp(-distances / (2 * sigma ** 2))  # 高斯相似度矩阵

    # 创建一个无向图，节点是像素，边是相似度（权重）
    G = nx.Graph()
    G.add_nodes_from(range(height * width))

    # 将每对像素的相似度作为边的权重
    for i in range(height * width):
        for j in range(i + 1, height * width):
            if W[i, j] > 0:
                G.add_edge(i, j, weight=W[i, j])

    # 使用NetworkX中的最大流算法来计算最小割
    # 创建源节点（s）和汇节点（t）
    s = height * width  # 源节点
    t = s + 1           # 汇节点

    # 将源节点和所有像素节点连接，目标是前景
    for i in range(height * width):
        G.add_edge(s, i, capacity=1)  # 源节点连接到每个像素节点，边的容量为1

    # 将每个像素节点连接到汇节点，目标是背景
    for i in range(height * width):
        G.add_edge(i, t, capacity=1)  # 每个像素节点连接到汇节点，边的容量为1

    # 计算最大流（即最小割）
    flow_value, partition = nx.minimum_cut(G, s, t, capacity='weight')

    # 获取前景和背景的节点集合
    reachable, non_reachable = partition

    # 创建分割结果图像
    segmented_image = np.zeros((height, width), dtype=int)
    for i in range(height * width):
        if i in reachable:
            segmented_image[i // width, i % width] = 1  # 前景为1
        else:
            segmented_image[i // width, i % width] = 0  # 背景为0

    return segmented_image


# 读取图像
image = io.imread('F:/DIP/experiment6/origin.png')

# 进行最小割分割
segmented_image = min_cut_segmentation(image)

# 可视化分割结果
plt.imshow(segmented_image, cmap='gray')
plt.title('Minimum Cut Segmentation')
plt.show()
