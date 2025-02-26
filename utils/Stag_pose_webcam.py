import stag
import cv2

import numpy as np

# 初始化摄像头
capture = cv2.VideoCapture(2)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
capture.set(cv2.CAP_PROP_FPS, 60)


# 棋盘格点配置
objp = np.zeros((2 * 2, 3), np.float32)
grid = np.mgrid[0:2, 0:2].T.reshape(-1, 2)
grid_sorted = np.zeros_like(grid)
grid_sorted[:, 0] = grid[:, 0]
grid_sorted[:, 1] = grid[:, 1]
grid_sorted[2, :], grid_sorted[3, :] = grid[3, :], grid[2, :]

# 将重排后的坐标赋值给 objp
objp[:, :2] = grid_sorted
worldpoint = objp * 62  # 棋盘格的宽度为50mm

# 摄像机矩阵和畸变系数
mtx = np.array([[928.1353689 ,   0.        , 629.41869016],
       [  0.        , 926.48166578, 366.79901931],
       [  0.        ,   0.        ,   1.        ]], dtype=np.float32)
dist = np.array([[-2.28062362e-01,  2.86530924e-01, -2.66013789e-04, -2.43976658e-03, -3.84812709e-01]], dtype=np.float32)

def rotation_matrix_to_quaternion(R):
    q = np.empty((4,))
    t = np.trace(R)
    if t > 0.0:
        t = np.sqrt(t + 1.0)
        q[3] = 0.5 * t
        t = 0.5 / t
        q[0] = (R[2, 1] - R[1, 2]) * t
        q[1] = (R[0, 2] - R[2, 0]) * t
        q[2] = (R[1, 0] - R[0, 1]) * t
    else:
        i = 0
        if R[1, 1] > R[0, 0]:
            i = 1
        if R[2, 2] > R[i, i]:
            i = 2
        j = (i + 1) % 3
        k = (j + 1) % 3
        t = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (R[k, j] - R[j, k]) * t
        q[j] = (R[j, i] + R[i, j]) * t
        q[k] = (R[k, i] + R[i, k]) * t
    return q


while True:
    # 调用摄像机
    ref, frame = capture.read()
    if not ref:
        continue

    # 复制一份原始帧，保持原始图像不变
    frame_with_axes = frame.copy()

    (corners, ids, rejected_corners) = stag.detectMarkers(frame, 17)
    print(corners)

    if len(ids) >= 1:
        for i, id in enumerate(ids):
            _, rvec, tvec = cv2.solvePnP(worldpoint, corners[i], mtx, dist)
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # 在每个检测到的二维码上绘制坐标系
            cv2.drawFrameAxes(frame_with_axes, mtx, dist, rvec, tvec, 50)  # 50是坐标轴的长度，单位像素

    # draw detected markers with ids
    stag.drawDetectedMarkers(frame, corners, ids)

    # draw rejected quads without ids with different color
    # stag.drawDetectedMarkers(frame, rejected_corners, border_color=(255, 0, 0))

    # 显示两个不同的窗口
    cv2.imshow("Camera Stream with Axes", frame_with_axes)  # 显示带坐标系的窗口
    cv2.imshow("Camera Stream with Markers", frame)  # 显示带标记的窗口

    # 检测用户按键，按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
capture.release()
cv2.destroyAllWindows()