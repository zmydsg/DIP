import numpy as np
import cv2
import maxflow

def build_graph(image, scale_factor=1.0):
    """
    构建稀疏图以用于最小割
    :param image: 灰度图像
    :param scale_factor: 图像缩放比例，用于多尺度处理
    :return: 图、节点 ID、缩放后的图像
    """
    # 缩放图像
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    
    # 构建图
    g = maxflow.Graph[float]()
    h, w = scaled_image.shape
    node_ids = g.add_nodes(h * w)
    
    # 定义邻接关系
    for y in range(h):
        for x in range(w):
            node = y * w + x
            if x + 1 < w:  # 横向边
                right_node = y * w + (x + 1)
                weight = np.exp(-abs(int(scaled_image[y, x]) - int(scaled_image[y, x + 1])) / 10)
                g.add_edge(node, right_node, weight, weight)
            if y + 1 < h:  # 纵向边
                bottom_node = (y + 1) * w + x
                weight = np.exp(-abs(int(scaled_image[y, x]) - int(scaled_image[y + 1, x])) / 10)
                g.add_edge(node, bottom_node, weight, weight)
    
    # 添加源点和汇点（简单分割：前景和背景假设）
    source = h * w  # 将源点放在图的最后一个节点
    sink = h * w + 1  # 将汇点放在源点之后

    g.add_nodes(2)  # 为源点和汇点添加节点
    
    for y in range(h):
        for x in range(w):
            node = y * w + x
            intensity = scaled_image[y, x]
            g.add_tedge(node, 255 - intensity, intensity)  # 假设亮度高的为前景
    
    # 添加源点和汇点连接
    g.add_tedge(source, 0, 0)  # 源点没有连接到任何节点
    g.add_tedge(sink, 0, 0)  # 汇点没有连接到任何节点
    
    return g, node_ids, scaled_image

def segment_image(image, coarse_scale=0.5):
    """
    多尺度图像分割
    :param image: 输入图像
    :param coarse_scale: 初始粗略分割的缩放比例
    :return: 分割后的图像
    """
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 粗尺度分割
    g, node_ids, scaled_image = build_graph(gray_image, scale_factor=coarse_scale)
    flow = g.maxflow()
    
    # 获取粗略分割结果
    h, w = scaled_image.shape
    coarse_segments = np.zeros_like(scaled_image, dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            node = y * w + x
            coarse_segments[y, x] = 255 if g.get_segment(node_ids[node]) == 1 else 0
    
    # 放大到原始图像大小
    fine_segments = cv2.resize(coarse_segments, (gray_image.shape[1], gray_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 在原始分辨率上细化分割
    g, node_ids, _ = build_graph(gray_image)
    flow = g.maxflow()
    
    h, w = gray_image.shape
    final_segments = np.zeros_like(gray_image, dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            node = y * w + x
            final_segments[y, x] = 255 if g.get_segment(node_ids[node]) == 1 else 0
    
    return final_segments

# 加载图像
input_image = cv2.imread('F:/DIP/experiment6/origin.png')

# 分割图像
segmented_image = segment_image(input_image, coarse_scale=0.5)

# 显示结果
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
