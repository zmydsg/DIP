import numpy as np
import cv2
import networkx as nx
from scipy.spatial.distance import euclidean

# 计算像素之间的亮度差异，作为边的权重
def compute_similarity(image):
    rows, cols = image.shape
    graph = nx.Graph()
    
    # 遍历图像的每个像素，创建图的节点
    for r in range(rows):
        for c in range(cols):
            graph.add_node((r, c), value=image[r, c])
    
    # 遍历相邻像素，计算亮度差异作为边的权重
    for r in range(rows):
        for c in range(cols):
            # 右边邻居
            if c + 1 < cols:
                weight = np.abs(int(image[r, c]) - int(image[r, c+1]))  # 基于亮度差异
                graph.add_edge((r, c), (r, c+1), weight=weight)
            # 下边邻居
            if r + 1 < rows:
                weight = np.abs(int(image[r, c]) - int(image[r+1, c]))  # 基于亮度差异
                graph.add_edge((r, c), (r+1, c), weight=weight)
    
    return graph

# 规范化割算法（简化版，实际应用中可以使用更复杂的优化方法）
def normalized_cut(graph):
    # 使用谱图分割方法，寻找图的最小割
    # 这里只是一个简单的示范，实际应用中可以用其他算法或库来实现
    cut_nodes = nx.algorithms.community.kernighan_lin_bisection(graph)
    return cut_nodes

# 基于图的亮度分割
def graph_based_image_segmentation(image_path):
    # 加载图像并转换为灰度图
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 计算图的相似度
    graph = compute_similarity(image)
    
    # 执行规范化割
    cut_nodes = normalized_cut(graph)
    
    # 创建分割后的图像
    segmented_image = np.zeros_like(image)
    for node in cut_nodes:
        r, c = node
        segmented_image[r, c] = 255  # 可以使用其他标记表示不同区域
    
    return segmented_image

# 示例使用
image_path = 'F:/DIP/experiment6/origin.png'  # 替换为你的图像路径
segmented_image = graph_based_image_segmentation(image_path)

# 显示结果
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
