import cv2
import numpy as np
import glob

def calibrate_camera(calib_images_path, pattern_size=(9, 6)):
    """
    进行相机标定，计算相机内参和畸变系数。

    :param calib_images_path: 包含棋盘格图像的路径，支持通配符，如 'calib_images/*.jpg'
    :param pattern_size: 棋盘格的内角数，如 (9, 6)
    :return: 相机矩阵、畸变系数、旋转向量和位移向量
    """
    # 准备棋盘格世界坐标
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

    objpoints = []  # 3D点在世界坐标系中的坐标
    imgpoints = []  # 2D点在图像平面的坐标

    # 获取所有棋盘格图像路径
    images = glob.glob(calib_images_path)

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"无法读取图像: {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            # 提高角点精度
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)

            # 可视化角点
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(100)
        else:
            print(f"未检测到棋盘格角点: {fname}")

    cv2.destroyAllWindows()

    if not objpoints or not imgpoints:
        raise ValueError("未检测到足够的棋盘格角点进行标定。")

    # 相机标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    if not ret:
        raise RuntimeError("相机标定失败。")

    print("相机矩阵:")
    print(camera_matrix)
    print("畸变系数:")
    print(dist_coeffs)

    return camera_matrix, dist_coeffs, rvecs, tvecs

def undistort_image(image_path, camera_matrix, dist_coeffs, save_path=None):
    """
    使用相机内参和畸变系数对图像进行去畸变。

    :param image_path: 待去畸变的图像路径
    :param camera_matrix: 相机内参矩阵
    :param dist_coeffs: 畸变系数
    :param save_path: 矫正后图像的保存路径（可选）
    :return: 矫正后的图像
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    h, w = img.shape[:2]

    # 获取优化后的相机矩阵
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # 去畸变
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # 裁剪图像
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]

    # 显示结果
    cv2.imshow("Original Image", img)
    cv2.imshow("Undistorted Image", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果（如果提供路径）
    if save_path:
        cv2.imwrite(save_path, undistorted_img)

    return undistorted_img

if __name__ == "__main__":
    # 标定参数
    calib_images_path = 'F:/DIP/experiment3/qipan.jpg'  # 替换为你的棋盘格图像路径
    pattern_size = (8, 7)  # 替换为你的棋盘格内角数

    # 执行相机标定
    camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(calib_images_path, pattern_size)

    # 矫正畸变图像
    distorted_image_path = 'F:/DIP/experiment3/jibian.png'  # 替换为你的畸变图像路径
    undistorted_image_save_path = 'F:/DIP\experiment3/undistorted_image.jpg'    #替换为矫正后图像的保存路径
    undistorted_img = undistort_image(distorted_image_path, camera_matrix, dist_coeffs, undistorted_image_save_path)
