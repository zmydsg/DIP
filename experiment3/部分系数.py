import numpy as np
import cv2
import glob

# 在标定时仅估计 k1 和 k2
dist_coeffs = np.zeros((2, 1))  # [k1, k2]

# 修改标定函数以仅估计径向畸变
def calibrate_single_image_radial(obj_points, img_points, camera_matrix, dist_coeffs):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1],
        camera_matrix, dist_coeffs,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS | 
              cv2.CALIB_FIX_PRINCIPAL_POINT | 
              cv2.CALIB_FIX_FOCAL_LENGTH | 
              cv2.CALIB_FIX_TANGENT_DIST | 
              cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | 
              cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
    )
    return mtx, dist

# 执行标定
mtx, dist = calibrate_single_image_radial(obj_points, img_points, camera_matrix, dist_coeffs)

print("校准后的相机矩阵：")
print(mtx)
print("校准后的畸变系数：")
print(dist)

# 畸变矫正
undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)

# 显示结果
cv2.imshow("Distorted Image", image)
cv2.imshow("Undistorted Image", undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存矫正后的图像
cv2.imwrite('undistorted_image.jpg', undistorted_image)
