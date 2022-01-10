import numpy as np
import cv2.cv2 as cv


# 使用高斯检测子计算
def myGaussBlur(image, ksize=5, sigma=1.5):
    """
    高斯滤波实现，高斯滤波内核的计算公式如下所示：
        G[i, j] = (1/(2*pi*sigma**2))*exp(-((i-k-1)**2 + (j-k-1)**2)/2*sigma**2)
    :param image: 输入原始图片
    :param ksize: 高斯滤波核大小，这里高斯滤波核长宽是相等的
    :param sigma: 高斯滤波的sigma参数
    :return: 输出滤波后的图片
    """
    # 计算高斯滤波器
    k = ksize // 2
    gaussian = np.zeros([ksize, ksize])
    for i in range(ksize):
        for j in range(ksize):
            gaussian[i, j] = np.exp(-((i - k) ** 2 + (j - k) ** 2) / (2 * sigma ** 2))
    gaussian /= 2 * np.pi * sigma ** 2
    # 归一化处理
    gaussian = gaussian / np.sum(gaussian)

    # 对图像进行高斯滤波
    W, H = image.shape
    new_image = np.zeros([W - k * 2, H - k * 2])

    for i in range(W - 2 * k):
        for j in range(H - 2 * k):
            new_image[i, j] = np.sum(image[i:i + ksize, j:j + ksize] * gaussian)

    new_image = np.uint8(new_image)

    return new_image


# 确定梯度幅值和方向
def get_gradient_and_direction(image):
    """
    计算图像的梯度幅值和方向
    使用Sobel算子来进行计算，其中：
         -1 0 1        -1 -2 -1
    Gx = -2 0 2   Gy =  0  0  0
         -1 0 1         1  2  1
    :param image: 输入图像
    :return:每个像素的梯度幅值, 每个梯度幅值方向
    """
    # 定义Gx与Gy
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    W, H = image.shape
    gradients = np.zeros([W - 2, H - 2])
    direction = np.zeros([W - 2, H - 2])

    # 计算梯度幅值和方向
    for i in range(W - 2):
        for j in range(H - 2):
            dx = np.sum(image[i:i + 3, j:j + 3] * Gx)
            dy = np.sum(image[i:i + 3, j:j + 3] * Gy)
            gradients[i, j] = np.sqrt(dx ** 2 + dy ** 2)
            if dx == 0:
                direction[i, j] = np.pi / 2
            else:
                direction[i, j] = np.arctan(dy / dx)

    gradients = np.uint8(gradients)

    return gradients, direction


# 非极大值抑制
def NMS(gradients, direction):
    """
    非极大值抑制
    :param gradients: 梯度幅值
    :param direction: 方向
    :return: 处理后的图像
    """
    W, H = gradients.shape
    nms = np.copy(gradients[1:-1, 1:-1])

    # 进行非极大值抑制
    for i in range(1, W - 1):
        for j in range(1, H - 1):
            theta = direction[i, j]
            weight = np.tan(theta)
            if theta > np.pi / 4:
                d1 = [0, 1]
                d2 = [1, 1]
                weight = 1 / weight
            elif theta >= 0:
                d1 = [1, 0]
                d2 = [1, 1]
            elif theta >= - np.pi / 4:
                d1 = [1, 0]
                d2 = [1, -1]
                weight *= -1
            else:
                d1 = [0, -1]
                d2 = [1, -1]
                weight = -1 / weight

            g1 = gradients[i + d1[0], j + d1[1]]
            g2 = gradients[i + d2[0], j + d2[1]]
            g3 = gradients[i - d1[0], j - d1[1]]
            g4 = gradients[i - d2[0], j - d2[1]]

            grade_count1 = g1 * weight + g2 * (1 - weight)
            grade_count2 = g3 * weight + g4 * (1 - weight)

            if grade_count1 > gradients[i, j] or grade_count2 > gradients[i, j]:
                nms[i - 1, j - 1] = 0

    return nms


# 边缘阈值和连接 (滞后法)
def double_threshold(nms, threshold1, threshold2):
    """
    双阈值选择与边缘连接
    通过假设两个阈值：低阈值TL和高阈值TH，并做如下的处理
    a.对于任意边缘像素低于TL的则丢弃
    b.对于任意边缘像素高于TH的则保留
    c.对于任意边缘像素值在TL与TH之间的，如果能通过边缘连接到一个像素大于TH而且边缘所有像素大于最小阈值TL的则保留，否则丢弃。
    :param nms: 输入的非最大化抑制图像
    :param threshold1: 低阈值
    :param threshold2: 高阈值
    :return: 二值图
    """
    visited = np.zeros_like(nms)
    output_image = nms.copy()
    W, H = output_image.shape

    # 内嵌一个深度搜索函数，处理情况b、c
    def dfs(i, j):
        if i >= W or i < 0 or j >= H or j < 0 or visited[i, j] == 1:
            return
        visited[i, j] = 1
        if output_image[i, j] > threshold1:
            output_image[i, j] = 255
            dfs(i - 1, j - 1)
            dfs(i - 1, j)
            dfs(i - 1, j + 1)
            dfs(i, j - 1)
            dfs(i, j + 1)
            dfs(i + 1, j - 1)
            dfs(i + 1, j)
            dfs(i + 1, j + 1)
        else:
            output_image[i, j] = 0

    for w in range(W):
        for h in range(H):
            if visited[w, h] == 1:
                continue
            if output_image[w, h] >= threshold2:  # 处理情况ｂ、c
                dfs(w, h)
            elif output_image[w, h] <= threshold1:  # 处理情况a
                output_image[w, h] = 0
                visited[w, h] = 1

    # 处理没有被访问到的像素点
    for w in range(W):
        for h in range(H):
            if visited[w, h] == 0:
                output_image[w, h] = 0

    return output_image


if __name__ == "__main__":
    # 读取图片
    image = cv.imread("data_set/lena512.bmp", 0)
    # 进行滤波，这里用的是手写的滤波方法
    smoothed_image = myGaussBlur(image)
    # smoothed_image = cv.GaussianBlur(image, (5, 5), 1.5)
    # 确定梯度幅值和方向
    gradients, direction = get_gradient_and_direction(smoothed_image)
    # 非极大值抑制
    nms = NMS(gradients, direction)
    # 边缘阈值和连接 (滞后法)
    output_image = double_threshold(nms, 30, 100)

    # 使用标准算法进行计算并且比较
    canny = cv.Canny(smoothed_image, 75, 150)
    # 两张图片进行与运算，计算不同的像素点数目和占比
    # 如何解决两张图片分辨率不对应的问题？(不可以resize！)
    # canny = cv.resize(canny, (500, 500))
    # output_image = cv.resize(output_image, (500, 500))
    canny = canny[4:504, 4:504]
    output_image = output_image[2:502, 2:502]
    result = canny ^ output_image
    total = np.count_nonzero(result)
    acc = total / np.count_nonzero(output_image)
    print("不同的像素点数:\t", total)
    print("边缘检测结果的非零像素点数:\t", np.count_nonzero(output_image))

    # 显示最后的结果
    cv.imshow("raw image", image)
    cv.imshow("Gauss Blur", smoothed_image)
    cv.imshow("gradients", gradients)
    cv.imshow("nms", nms)
    cv.imshow("my canny", output_image)
    cv.imshow("opencv canny", canny)
    cv.waitKey(0)
