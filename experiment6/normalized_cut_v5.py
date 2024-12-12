import cv2
import numpy as np
from skimage import img_as_float
from skimage.future import graph
from skimage.segmentation import slic, mark_boundaries
from skimage.color import rgb2gray, gray2rgb

# 1. 读取图像与预处理
img = cv2.imread("path_to_image.jpg", cv2.IMREAD_COLOR)
# 转换为RGB格式（scikit-image通常使用RGB而非BGR）
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 转灰度
gray = rgb2gray(img)

# 双边滤波自适应平滑
# d: 滤波邻域直径，sigmaColor/sigmaSpace控制平滑强度
smoothed = cv2.bilateralFilter((gray*255).astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=75)
smoothed = smoothed.astype(np.float32)/255.0

# CLAHE增强
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply((smoothed*255).astype(np.uint8))
clahe_img = clahe_img.astype(np.float32)/255.0

# 将增强图像转换为RGB（以便后续segmentation的处理）
enhanced_img = gray2rgb(clahe_img)

# 2. 建立图像金字塔
# 通常高斯金字塔通过cv2.pyrDown缩小图像尺寸以减小计算量
pyramid = [enhanced_img]
num_levels = 2  # 根据图像大小和计算量要求，自定义金字塔层数
for _ in range(num_levels):
    # pyrDown减半图像尺寸
    down = cv2.pyrDown(pyramid[-1])
    pyramid.append(down)

# 现在pyramid[-1]是最低分辨率图像

# 3. 在较低分辨率下进行过分割和Normalized Cut
low_res_img = pyramid[-1]
# 转为浮点型方便SLIC处理
low_res_img_float = img_as_float(low_res_img)

# 使用SLIC超像素分割（可根据需要调参）
# n_segments可根据金字塔层图像大小调整
segments = slic(low_res_img_float, n_segments=100, compactness=10, start_label=1)

# 基于分割结果构建区域邻接图(RAG)
rag = graph.rag_mean_color(low_res_img_float, segments)

# 使用Normalized Cut对RAG进行分割
# cut_normalized的threshold参数可调，可能需要尝试不同参数
labels = graph.cut_normalized(segments, rag, num_cuts=10)

# 此时labels为低分辨率下的分割标签图像

# 4. 将低分辨率分割结果向上映射（如果需要多级精细化）
# 简单方法：直接上采样
for i in range(num_levels):
    # 每次向上一层映射需要上采样labels
    h, w, _ = pyramid[num_levels - 1 - i].shape
    labels = cv2.resize(labels.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

final_labels = labels  # final_labels为原始图像分辨率下的分割标签结果

# 5. 可视化结果
# 为展示，将标签转换为不同颜色显示
def label2color(label_img):
    # 为每个label生成随机颜色
    labels_unique = np.unique(label_img)
    output = np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)
    np.random.seed(0)
    for lbl in labels_unique:
        color = np.random.randint(0, 255, 3)
        output[label_img==lbl] = color
    return output

segmented_color = label2color(final_labels)

# 显示结果（需要在本地环境下运行）
cv2.imshow("Original", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.imshow("Enhanced Image", cv2.cvtColor((enhanced_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
cv2.imshow("Segmented (Ncut)", cv2.cvtColor(segmented_color, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存最终分割结果
cv2.imwrite("F:/DIP/experiment6/ncut_segmented_result.png", cv2.cvtColor(segmented_color, cv2.COLOR_RGB2BGR))
