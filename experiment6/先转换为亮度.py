import cv2
import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def adjust_brightness(image, beta=50):
    """
    调整图像亮度。
    beta > 0 增加亮度，beta < 0 减少亮度
    """
    adjusted = cv2.convertScaleAbs(image, alpha=1, beta=beta)
    return adjusted

def extract_features(image):
    """
    提取图像特征，包括颜色和空间位置。
    """
    # 获取图像的高、宽
    h, w, c = image.shape
    # 创建网格坐标
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    # 将特征展平成二维数组
    features = np.zeros((h * w, 5))
    features[:, 0:3] = image.reshape(-1, 3)  # 颜色特征 (B, G, R)
    features[:, 3] = x.reshape(-1) / w  # 归一化x坐标
    features[:, 4] = y.reshape(-1) / h  # 归一化y坐标
    
    # 标准化特征以提高聚类效果
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    return features

def normalized_cut_segmentation(image, n_clusters=2, n_neighbors=20):
    """
    使用归一化切（Spectral Clustering）进行图像分割。
    
    参数:
    - image: 输入图像
    - n_clusters: 聚类数量
    - n_neighbors: 最近邻数量
    """
    features = extract_features(image)
    # 使用谱聚类进行分割
    spectral = SpectralClustering(n_clusters=n_clusters, 
                                  eigen_solver='arpack',
                                  affinity='nearest_neighbors',
                                  n_neighbors=n_neighbors,
                                  assign_labels='kmeans',
                                  random_state=42)
    labels = spectral.fit_predict(features)
    # 将标签重塑为图像尺寸
    segmented = labels.reshape(image.shape[:2])
    return segmented

def main():
    # 读取图像
    image_path = 'F:/DIP/experiment6/origin.png'  # 替换为你的图像路径
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像，请检查路径是否正确。")
        return
    
    # 可选：缩放图像以减少计算量
    scale_percent = 50  # 缩放到50%，根据需要调整
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    # 调整亮度
    bright_image = adjust_brightness(image, beta=50)  # 增加亮度
    
    # 图像分割
    n_clusters = 4  # 分割成区域，可以根据需要调整
    n_neighbors = 30  # 增加邻居数量以提高连通性
    segmented = normalized_cut_segmentation(bright_image, n_clusters=n_clusters, n_neighbors=n_neighbors)
    
    # 映射分割结果到颜色
    segmented_color = np.zeros_like(image)
    colors = [
        [0, 0, 0],        # 黑色
        [255, 0, 0],      # 蓝色
        [0, 255, 0],      # 绿色
        [0, 0, 255],      # 红色
        [255, 255, 0],    # 青色
        [255, 0, 255],    # 品红
        [0, 255, 255],    # 黄色
        [255, 255, 255]   # 白色
    ]
    for i in range(n_clusters):
        segmented_color[segmented == i] = colors[i % len(colors)]
    
    # 显示结果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.title('原始图像')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('亮度调整后的图像')
    plt.imshow(cv2.cvtColor(bright_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('归一化切分割结果')
    plt.imshow(segmented_color)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 保存结果
    cv2.imwrite('F:/DIP/experiment6/liangdu_segmented_image.jpg', segmented_color)
    print("分割结果已保存为 'segmented_image.jpg'")

if __name__ == "__main__":
    main()
