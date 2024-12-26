import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_edge_baseline(image_path, output_path=None, threshold_value=100):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像，请检查路径是否正确。")
        return
    
    # 2. 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 可选：自适应直方图均衡化（CLAHE），增强对比度
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(6,6))
    enhanced = clahe.apply(gray)

    # 4. 自适应中值滤波，去除噪声
    enhanced = cv2.medianBlur(enhanced, 5)

    # 4. Sobel 算子计算梯度
    #    - dx=1, dy=0 表示 x 方向的梯度
    #    - dx=0, dy=1 表示 y 方向的梯度
    sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)

    # 5. 取绝对值并转换为 8 位图像
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    # 6. 合并 x 和 y 两个方向梯度（也可以只取其中一个方向）
    sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

    # 7. 二值化处理，让轮廓为白色，其他全黑
    #    方案A: 使用固定阈值
    #    ret, binary_sobel = cv2.threshold(sobel_combined, threshold_value, 255, cv2.THRESH_BINARY)
    #
    #    方案B: 使用 Otsu 自适应阈值
    ret, binary_sobel = cv2.threshold(sobel_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 8. 显示结果
    #    使用 matplotlib 显示多个结果对比
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title('原始图像')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('增强后 (CLAHE)')
    plt.imshow(enhanced, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Sobel 二值边缘')
    plt.imshow(binary_sobel, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 9. 保存结果（如果指定了 output_path）
    if output_path:
        cv2.imwrite(output_path, binary_sobel)
        print(f"Sobel 二值边缘检测结果已保存到: {output_path}")

if __name__ == "__main__":
    # 示例用法
    input_image_path = 'F:\DIP\experiment5\origin.png'  # 替换为你的图像路径
    output_image_path = 'F:\DIP\DIP_finalpaper/sobelbaseline11.png'  # 替换为想要保存的路径
    sobel_edge_baseline(input_image_path, output_image_path, threshold_value=100)
