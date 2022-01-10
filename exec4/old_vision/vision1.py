"""
TODO
 要求：
    1.给出你用手机拍摄两张有重叠视野的照片（建议两幅图像打光不同，以体现第五步融合的效果）。
    2.用sift特征或者任意你熟悉的特征进行匹配。（附特征点图）
    3.用RANSAC方法去除误匹配点对。（附筛除后的特征点图）
    4.给出变换矩阵（仿射模型或投射模型都可以，需注明），并完成拼接（前向映射或反向映射都可以，需注明），给出拼接后的图像。
    5.对拼接后的图像进行融合（任意你喜欢的融合方法），附融合后的图像。
 注意：
    尽量描述清楚每一步的思路，并附上对应代码。
    完成任意四步就可以拿到满分（即可以不筛除离群点，或不进行融合）。
    网上代码较多，可以参考。
"""
import time

import cv2.cv2 as cv
import numpy as np


# 1.给出你用手机拍摄两张有重叠视野的照片（建议两幅图像打光不同，以体现第五步融合的效果）。
def load_image(img1_path, img2_path):
    """
    加载两张图片，并进行预处理
    :param img1_path: 图片1的路径
    :param img2_path: 图片2的路径
    :return: 处理后的图片1与图片2
    """
    # ================ 图片加载 ================
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)
    # ================ 分辨率调整 ================
    rate = 600 / img1.shape[1]
    img1 = cv.resize(img1, (int(rate * img1.shape[1]), int(rate * img1.shape[0])))
    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))
    # FIXME:
    #  - 使用边界填充：特征点匹配的结果图带有很大的黑域
    #  - 不使用边界填充：最后图像拼接和融合不方便，还是需要新建一张更大的图或者提前算好图的大小
    # ================ 边界填充 ================
    top, bot, left, right = 250, 250, 250, 250
    img1 = cv.copyMakeBorder(img1, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    img2 = cv.copyMakeBorder(img2, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))

    return img1, img2


# 2.用sift特征或者任意你熟悉的特征进行匹配。（附特征点图）
def match_feather_point(img1, img2):
    """
    计算特征点并进行匹配
    :param img1: 第一张图片
    :param img2: 第二张图片
    :return: 匹配的特征点
    """
    # ================ 计算SIFT特征 ================
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # ================ FLANN特征匹配 ================
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    return kp1, kp2, matches


# 3.用RANSAC方法去除误匹配点对。（附筛除后的特征点图）
# 4.给出变换矩阵（仿射模型或投射模型都可以，需注明），并且进行图像配准，完成拼接（前向映射或反向映射都可以，需注明），给出拼接后的图像。
def get_good_match(img1, img2, kp1, kp2, matches):
    """"""
    # TODO 使用RANSAC方法去除错误匹配点对
    matchesMask = [[0, 0] for _ in range(len(matches))]
    good_match = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good_match.append(m)
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    MIN_MATCH_COUNT = 10
    if len(good_match) < MIN_MATCH_COUNT:
        print("Not enough matches are found - {}/{}".format(len(good_match), MIN_MATCH_COUNT))
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    img2 = cv.warpPerspective(img2, np.array(M), (img2.shape[1], img2.shape[0]),
                              flags=cv.WARP_INVERSE_MAP)

    # 拷贝拼接(非常非常简单的拷贝拼接，效果不好，也不好拓展！)
    dst = img1.copy()
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            pix = img2[i, j]
            if pix.any():
                dst[i, j] = pix

    return img1, img2, img3, dst, M


# 5.对拼接后的图像进行融合（任意你喜欢的融合方法），附融合后的图像。
def blend_image(srcImg, warpImg):
    """"""
    rows, cols = srcImg.shape[:2]
    # 找到左右重叠区域
    for col in range(0, cols):
        if srcImg[:, col].any() and warpImg[:, col].any():
            left = col
            break
    for col in range(cols - 1, 0, -1):
        if srcImg[:, col].any() and warpImg[:, col].any():
            right = col
            break

    res = np.zeros([rows, cols, 3], np.uint8)
    alpha = np.zeros((rows, right - left, 3), dtype=np.float)
    for row in range(0, rows):
        for col in range(left, right):
            if not srcImg[row, col].any():  # src不存在
                alpha[row, col - left, :] = 0
            elif not warpImg[row, col].any():  # warp_img 不存在
                alpha[row, col - left, :] = 1
            else:  # src 和warp都存在
                srcImgLen = float(abs(col - left))
                testImgLen = float(abs(col - right))
                alpha[row, col - left, :] = testImgLen / (srcImgLen + testImgLen)

    res[:, :left] = srcImg[:, :left]
    res[:, right:] = warpImg[:, right:]
    res[:, left:right] = np.clip(srcImg[:, left:right] * alpha + warpImg[:, left:right] * (np.ones_like(alpha) - alpha),
                                 0, 255)

    return res


def main_process(path1='./images/testA_1.jpg', path2='./images/testA_2.jpg'):
    """"""
    srcImg, warpImg = load_image(path1, path2)

    start = time.time()
    kp1, kp2, matches = match_feather_point(srcImg, warpImg)
    end = time.time()
    print('特征点计算以及匹配的时间：', end - start)

    start = time.time()
    img1, img2, img3, dst, M = get_good_match(srcImg, warpImg, kp1, kp2, matches)
    end = time.time()
    print('去除误匹配点、计算变换矩阵并进行拼接的时间：', end - start)

    start = time.time()
    res = blend_image(img1, img2)
    end = time.time()
    print('图像融合时间：', end - start)

    print("变换矩阵", M)
    cv.imshow("splicing", dst)
    cv.imshow("sift_match", img3)
    cv.imshow("result", res)
    cv.waitKey()


if __name__ == "__main__":
    path1 = "./images/testD_2.jpg"
    path2 = "./images/testX_1.jpg"
    main_process(path1, path2)
