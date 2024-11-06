import cv2
import numpy as np
import glob

# 配置棋盘格尺寸 (例如 9x6 内部角点数量)
CHECKERBOARD = (10, 9)

# 存储检测到的3D和2D角点
objpoints = []  # 3D 点
imgpoints = []  # 2D 点

# 准备棋盘格的3D坐标（假设每个格子的大小为1单位）
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 存储需要手动标注的图像
manual_images = []

# 读取所有标定图像
images = glob.glob('F:\DIP\experiment3\qipan.png')  # 修改为你的标定图像路径

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 尝试自动检测棋盘格角点
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    )

    if ret:
        # 细化角点位置
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        imgpoints.append(corners2)
        objpoints.append(objp)

        # 可视化角点检测
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(500)
    else:
        # 如果检测失败，记录需要手动标注的图像
        manual_images.append((fname, img))

cv2.destroyAllWindows()

# 手动标注角点函数
def manual_annotation(image):
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Manual Annotation', image)

    cv2.imshow('Manual Annotation', image)
    cv2.setMouseCallback('Manual Annotation', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.array(points, dtype=np.float32)

# 对需要手动标注的图像逐个处理
for fname, img in manual_images:
    print(f"手动标注图像: {fname}")
    manual_corners = manual_annotation(img)

    if len(manual_corners) == CHECKERBOARD[0] * CHECKERBOARD[1]:
        imgpoints.append(manual_corners)
        objpoints.append(objp)
    else:
        print(f"图像 {fname} 的标注角点数量不足，请重新标注。")

# 计算相机内参和畸变系数
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 畸变校正示例
img = cv2.imread('F:\DIP\experiment3\jibian.png')  # 替换为你的畸变图像路径
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# 矫正图像
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# 裁剪掉黑边
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# 保存和显示结果
cv2.imwrite('calib_result.jpg', dst)
cv2.imshow('Undistorted', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
