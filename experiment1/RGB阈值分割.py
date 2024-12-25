import cv2
import numpy as np

def extract_color_region(image, color='yellow'):
    # 将输入图像转换为RGB格式
    img = cv2.imread(image)
    
    if img is None:
        print("无法读取图像")
        return

    # 定义黄色或蓝色的RGB阈值范围
    if color == 'yellow':
        # 选择黄色范围: 在RGB中，黄色是红色和绿色通道较强
        lower_bound = np.array([100, 100, 0])  # RGB最小值
        upper_bound = np.array([255, 255, 100])  # RGB最大值
    elif color == 'blue':
        # 选择蓝色范围: 蓝色在RGB中蓝色通道较强
        lower_bound = np.array([0, 0, 100])  # RGB最小值
        upper_bound = np.array([100, 100, 255])  # RGB最大值
    else:
        print("颜色无效，选择 'yellow' 或 'blue'")
        return

    # 进行阈值分割，提取指定颜色的区域
    mask = cv2.inRange(img, lower_bound, upper_bound)

    # 提取指定颜色区域的结果
    result = cv2.bitwise_and(img, img, mask=mask)

    # 显示结果
    cv2.imshow(f'{color.capitalize()} Extracted', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 示例用法
image_path = 'path_to_your_image.jpg'
extract_color_region(image_path, color='yellow')  # 提取黄色区域
# extract_color_region(image_path, color='blue')  # 提取蓝色区域
