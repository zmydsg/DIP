import numpy as np
import cv2
import glob

# 设置棋盘格的大小
chessboard_size = (8, 7)  # 棋盘的内部交点数量，例如7x7

# 准备棋盘格点的世界坐标系的对象点，假设棋盘的Z=0
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 用于存储所有棋盘图像的对象点和图像点
objpoints = []  # 3D点
imgpoints = []  # 2D点

# 读取棋盘图像
images = ['F:\DIP\experiment3\qipan.png']  # 加载你上传的棋盘图片路径


for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load image at path: {fname}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘角点
    # 调用 findChessboardCorners，并加入标志位提高检测鲁棒性
    ret, corners = cv2.findChessboardCorners(
    gray, chessboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
    # 如果找到，添加对象点和图像点
    if ret:
        print(f"Found corners in {fname}")
        objpoints.append(objp)
        imgpoints.append(corners)

        # 可视化角点
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(500)
        #保存一下
        cv2.imwrite('F:\DIP\experiment3\jiaodian.png', img)
        print("图像保存成功！")
    else:
        print(f"Failed to find corners in {fname}")




cv2.destroyAllWindows()

# 标定相机
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 打印相机的内参矩阵和畸变系数
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# 读取畸变的日历钟表图像
distorted_img = cv2.imread('F:\DIP\experiment3\jibian.png')

# 对图像进行畸变矫正
h, w = distorted_img.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

# 使用校正矩阵对图像进行去畸变
undistorted_img = cv2.undistort(distorted_img, camera_matrix, dist_coeffs, None, new_camera_matrix)


# 显示原图和去畸变后的图像
cv2.imshow("Original Image", distorted_img)
cv2.imshow("Undistorted Image", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存去畸变后的图像
# cv2.imwrite('F:\DIP\experiment3\jiaozheng.png', undistorted_img)
