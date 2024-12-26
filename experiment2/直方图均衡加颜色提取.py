import cv2
import numpy as np

# 读取图像
image = cv2.imread('F:\DIP\experiment1\HSV\original2.png')  # 替换为你的图像路径
# 将BGR转换为HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 分离HSV通道
h, s, v = cv2.split(hsv)

# 对V通道进行直方图均衡化
v_eq = cv2.equalizeHist(v)

# 合并均衡化后的V通道
hsv_eq = cv2.merge([h, s, v_eq])

# 转换回BGR颜色空间
image_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
# 将均衡化后的图像转换为HSV
hsv_eq = cv2.cvtColor(image_eq, cv2.COLOR_BGR2HSV)

# 定义黄色的HSV范围
# 黄色的Hue大约在20到30之间（范围可根据需要调整）
lower_yellow = np.array([15, 100, 100])  # 下界
upper_yellow = np.array([35, 255, 255])  # 上界

# 创建黄色掩膜
mask_yellow = cv2.inRange(hsv_eq, lower_yellow, upper_yellow)

# 可选：使用形态学操作（如膨胀）来扩大黄色区域
kernel = np.ones((5,5), np.uint8)
mask_yellow = cv2.dilate(mask_yellow, kernel, iterations=1)

# 将掩膜应用到均衡化后的图像上
yellow_extracted = cv2.bitwise_and(image_eq, image_eq, mask=mask_yellow)


# 显示原始图像、均衡化图像、掩膜和提取的黄色区域
cv2.imshow('Original Image', image)
cv2.imshow('Histogram Equalized Image', image_eq)
cv2.imshow('Yellow Mask', mask_yellow)
cv2.imshow('Yellow Extracted', yellow_extracted)

# 等待按键然后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

# 或者保存结果
cv2.imwrite('F:\DIP\experiment2\yellow_extracted3.jpg', yellow_extracted)
cv2.imwrite('F:\DIP\experiment2\mask_yellow3.jpg', mask_yellow)
cv2.imwrite('F:\DIP\experiment2\image_eq3.jpg', image_eq)