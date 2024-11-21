import cv2
import numpy as np

def edge_detection(input_image_path, output_canny_path, output_log_path):
    # 1. 读取图像
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read the image at {input_image_path}")
        return

    # 2. 自适应直方图均衡化（CLAHE）降噪
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(6, 6))
    equalized_image = clahe.apply(image)

    # 3. 使用Canny算子进行边缘检测
    # 首先进行高斯滤波以减少噪声
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 2.0)
    # 使用Canny进行边缘检测
    canny_edges = cv2.Canny(blurred_image, threshold1=50, threshold2=250)
    
    # 使用高斯-拉普拉斯（LoG）进行边缘检测
    # 先进行高斯模糊，再去噪，然后计算拉普拉斯
    log_blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 1.4)
    log_denoised_image = cv2.fastNlMeansDenoising(log_blurred_image, None, 30, 7, 21)
    log_edges = cv2.Laplacian(log_denoised_image, cv2.CV_64F)
    log_edges = cv2.convertScaleAbs(log_edges, alpha=2.0, beta=50)  # 调整对比度和亮度以减少图像过暗的问题


    # 5. 保存结果到指定路径
    cv2.imwrite(output_canny_path, canny_edges)
    cv2.imwrite(output_log_path, log_edges)

    print(f"Canny edges saved at: {output_canny_path}")
    print(f"LoG edges saved at: {output_log_path}")

# 示例用法
input_image_path = 'F:/DIP/experiment5/origin.png'  # 替换为你的输入图像路径
output_canny_path = 'F:/DIP/experiment5/cannydege.png'  # Canny边缘检测结果保存路径
output_log_path = 'F:/DIP/experiment5/LOGdege.png'  # LoG边缘检测结果保存路径

edge_detection(input_image_path, output_canny_path, output_log_path)
