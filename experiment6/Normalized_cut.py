import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from PIL import Image

# 1. 构建加权图
def build_graph(data, sigma=1.0, k=10):
    """
    构建加权邻接图，采用 k 近邻加权方式。
    data: 数据点数组 (n_samples, n_features)
    sigma: 高斯核参数
    k: 每个点的近邻数
    """
    n = data.shape[0]
    # 计算点对的高斯核相似度
    affinity_matrix = rbf_kernel(data, gamma=1.0 / (2 * sigma**2))
    
    # 只保留 k 近邻
    for i in range(n):
        row = affinity_matrix[i, :]
        sorted_indices = np.argsort(row)[::-1]
        row[sorted_indices[k+1:]] = 0  # 仅保留 k 个近邻
        affinity_matrix[i, :] = row

    affinity_matrix = (affinity_matrix + affinity_matrix.T) / 2  # 保证对称性
    return csr_matrix(affinity_matrix)

# 2. 计算归一化拉普拉斯矩阵
def normalized_laplacian(W):
    """
    计算归一化图拉普拉斯矩阵
    W: 加权邻接矩阵 (稀疏矩阵格式)
    """
    d = np.array(W.sum(axis=1)).flatten()
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    L = np.eye(W.shape[0]) - D_inv_sqrt @ W.toarray() @ D_inv_sqrt
    return csr_matrix(L)

# 3. 图分割（特征值分解）
def spectral_clustering(L, n_clusters=2):
    """
    使用特征值分解进行图分割
    L: 归一化图拉普拉斯矩阵
    n_clusters: 分割的子图数量
    """
    # 提取前 n_clusters 个最小特征值和特征向量
    _, eigvecs = eigsh(L, k=n_clusters, which='SM', tol=1e-2)
    return eigvecs

# 4. 分配数据到簇
def assign_clusters(eigvecs):
    """
    根据特征向量将数据分配到簇
    eigvecs: 特征向量矩阵
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, n_init=10)
    labels = kmeans.fit_predict(eigvecs)
    return labels

# 5. 可视化分割结果
def plot_image_segmentation(image, labels):
    """
    将聚类结果映射回图片
    image: 输入图片 (H, W, C)
    labels: 每个像素的簇标签
    """
    h, w, c = image.shape
    segmented_image = np.zeros((h, w, c), dtype=np.uint8)
    unique_labels = np.unique(labels)
    colors = [np.random.randint(0, 255, 3) for _ in unique_labels]

    # 分配每个像素对应的颜色
    for label, color in zip(unique_labels, colors):
        segmented_image[labels.reshape(h, w) == label] = color

    plt.imshow(segmented_image)
    plt.title("Image Segmentation Results")
    plt.axis("off")
    plt.show()

# 6. 图片处理函数
def process_image(image_path, sigma=0.1, k=10, n_clusters=2):
    """
    对图片进行分割处理
    image_path: 图片路径
    sigma: 高斯核参数
    k: 每个像素的近邻数
    n_clusters: 分割的子图数量
    """
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    h, w, c = image.shape

    # 构建点云数据 (像素坐标 + 颜色值)
    coords = np.indices((h, w)).reshape(2, -1).T
    colors = image.reshape(-1, 3)
    data = np.hstack((coords, colors))  # (n_pixels, 5)

    # 构建加权图
    W = build_graph(data, sigma=sigma, k=k)
    
    # 计算归一化拉普拉斯矩阵
    L = normalized_laplacian(W)
    
    # 特征分解 (优化计算特征值数量)
    eigvecs = spectral_clustering(L, n_clusters=n_clusters)
    
    # 分配到簇
    labels = assign_clusters(eigvecs)

    # 可视化结果
    plot_image_segmentation(image, labels)

# 7. 测试运行
if __name__ == "__main__":
    # 替换为本地图片路径
    image_path = "F:/DIP/experiment6/origin.png"  # 图片路径
    process_image(image_path, sigma=5.0, k=10, n_clusters=2)
