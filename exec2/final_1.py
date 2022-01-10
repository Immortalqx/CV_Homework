"""
任务一：基本处理Harris角点检测
数据：棋盘图片
要求：自己写函数实现Harris角点检测子，设置不同参数，比较检测结果
     其中  边缘检测子：sobel检测子
          参数包括，窗口大小和检测阈值
"""
import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt


# 这个函数是用OpenCV实现的Harris角点检测
def harris_demo(filepath):
    if filepath is None:
        return
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    # 图像转换为float32
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)  # 图像膨胀
    img[dst > 0.01 * dst.max()] = [0, 0, 255]  # 角点位置用红色标记
    # 这里的打分值以大于0.01×dst中最大值为边界
    return img


# 这里是我自己的Harris角点检测实现
def my_cornerHarris(img, sigmaX=2, ksize=3, k=0.04, threshold=0.01, WITH_NMS=False):
    # 1、使用Sobel计算像素点x,y方向的梯度
    h, w = img.shape[:2]
    # Sobel函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，所以Sobel建立的图像位数不够，会有截断。因此要使用16位有符号的数据类型，即cv2.CV_16S。
    grad = np.zeros((h, w, 2), dtype=np.float32)
    grad[:, :, 0] = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    grad[:, :, 1] = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    # 2、计算Ix^2,Iy^2,Ix*Iy
    m = np.zeros((h, w, 3), dtype=np.float32)
    m[:, :, 0] = grad[:, :, 0] ** 2
    m[:, :, 1] = grad[:, :, 1] ** 2
    m[:, :, 2] = grad[:, :, 0] * grad[:, :, 1]

    # 3、利用高斯函数对Ix^2,Iy^2,Ix*Iy进行滤波
    m[:, :, 0] = cv2.GaussianBlur(m[:, :, 0], ksize=(ksize, ksize), sigmaX=sigmaX)
    m[:, :, 1] = cv2.GaussianBlur(m[:, :, 1], ksize=(ksize, ksize), sigmaX=sigmaX)
    m[:, :, 2] = cv2.GaussianBlur(m[:, :, 2], ksize=(ksize, ksize), sigmaX=sigmaX)
    m = [np.array([[m[i, j, 0], m[i, j, 2]], [m[i, j, 2], m[i, j, 1]]]) for i in range(h) for j in range(w)]

    # 4、计算局部特征结果矩阵M的特征值和响应函数R(i,j)=det(M)-k(trace(M))^2  0.04<=k<=0.06
    D, T = list(map(np.linalg.det, m)), list(map(np.trace, m))
    R = np.array([d - k * t ** 2 for d, t in zip(D, T)])

    # 5、将计算出响应函数的值R进行非极大值抑制，滤除一些不是角点的点，同时要满足大于设定的阈值
    # 获取最大的R值
    R_max = np.max(R)
    # print(R_max)
    # print(np.min(R))
    R = R.reshape(h, w)
    corner = np.zeros_like(R, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if WITH_NMS:
                # 除了进行进行阈值检测 还对3x3邻域内非极大值进行抑制(导致角点很小，会看不清)
                if R[i, j] > R_max * threshold and R[i, j] == np.max(
                        R[max(0, i - 1):min(i + 2, h - 1), max(0, j - 1):min(j + 2, w - 1)]):
                    corner[i, j] = 255
            else:
                # 只进行阈值检测
                if R[i, j] > R_max * threshold:
                    corner[i, j] = 255
    return corner


# harris角点检测并可视化出来，除了中间的函数其余部分和harris_demo是一样的
def my_harris(filepath):
    img = cv2.imread(filepath)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = my_cornerHarris(gray_img, 2, 3, 0.04, WITH_NMS=True)
    # result = my_cornerHarris(gray_img, 2, 3, 0.04, WITH_NMS=False)
    dst = cv2.dilate(result, None)  # 图像膨胀
    img[dst > 0.01 * dst.max()] = [0, 0, 255]  # 角点位置用红色标记
    return img


if __name__ == "__main__":
    filepath = "check_board.png"

    demo_img = harris_demo(filepath)
    mine_img = my_harris(filepath)

    fig = plt.figure("Harris", figsize=(5, 3))
    ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[], title="opencv demo")
    ax.imshow(cv2.cvtColor(demo_img, cv2.COLOR_BGR2RGB))
    ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[], title="mine result")
    ax.imshow(cv2.cvtColor(mine_img, cv2.COLOR_BGR2RGB))
    plt.show()
