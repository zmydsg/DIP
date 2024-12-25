import cv2
import numpy as np

# 读取图像
image = cv2.imread('F:\DIP\experiment1\HSV\original2.png')

# 手动框定区域
roi = cv2.selectROI("Image", image, fromCenter=False, showCrosshair=True)

# 提取选定区域的图像
roi_image = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

# 计算选定区域的平均颜色
mean_color = cv2.mean(roi_image)[:3]

# 定义颜色阈值范围
tolerance = 30
lower_bound = np.array([max(0, mean_color[0] - tolerance), max(0, mean_color[1] - tolerance), max(0, mean_color[2] - tolerance)])
upper_bound = np.array([min(255, mean_color[0] + tolerance), min(255, mean_color[1] + tolerance), min(255, mean_color[2] + tolerance)])

# 在整个图像中寻找匹配颜色的区域
mask = cv2.inRange(image, lower_bound, upper_bound)

# 使用mask在原图上提取匹配的区域
result = cv2.bitwise_and(image, image, mask=mask)

# cv2.imwrite('F:/DIP/experiment1/result.png', result)
# 显示结果
cv2.imshow('Matched Colors', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
