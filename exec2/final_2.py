"""
任务二：SIFT算法
数据：山脉图片1和图片2
要求：使用SIFT算法实现两幅山脉图片的匹配
     1.尝试自己写函数实现SIFT算法
     2.调节算法参数，进行匹配结果对比
     3.与调用库函数结果进行对比
"""
import cv2.cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

from SIFT import my_SIFT


def sift_demo():
    # 1) 以灰度图的形式读入图片
    psd_img_1 = cv2.imread('mountain_1.png', 0)
    psd_img_2 = cv2.imread('mountain_2.png', 0)

    # 2) SIFT特征计算
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT_create()

    psd_kp1, psd_des1 = sift.detectAndCompute(psd_img_1, None)
    psd_kp2, psd_des2 = sift.detectAndCompute(psd_img_2, None)

    # 3) Flann特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(psd_des1, psd_des2, k=2)
    goodMatch = []
    for m, n in matches:
        # goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
        if m.distance < 0.50 * n.distance:
            goodMatch.append(m)
    # 增加一个维度
    goodMatch = np.expand_dims(goodMatch, 1)

    img_out = cv2.drawMatchesKnn(psd_img_1, psd_kp1, psd_img_2, psd_kp2, goodMatch[:15], None, flags=2)

    return img_out


if __name__ == "__main__":
    demo_img = sift_demo()
    mine_img = my_SIFT()

    fig = plt.figure("SIFT", figsize=(5, 3))
    ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[], title="opencv demo")
    ax.imshow(cv2.cvtColor(demo_img, cv2.COLOR_BGR2RGB))
    ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[], title="mine result")
    ax.imshow(cv2.cvtColor(mine_img, cv2.COLOR_BGR2RGB))
    plt.show()
