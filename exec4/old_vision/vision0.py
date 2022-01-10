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
import cv2.cv2 as cv

img1_path = ""
img2_path = ""


def load_image():
    """
    加载两张图片，并进行预处理
    :return: img1,img2
    """
    # ================ 图片加载部分 ================
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)
    # ============== 分辨率调整与滤波 ==============
    # TODO
    img1_size = img1.shape[:2]
    img2 = cv.resize(img2, (img1_size[1], img1_size[0]))
    # ============================================
    return img1, img2


def match_feather_point():
    """"""
    pass


def remove_wrong_match():
    """"""
    pass


def stitched_image():
    """"""
    pass


def fusion():
    """"""
    pass


def main_process():
    """"""
    pass


if __name__ == "__main__":
    main_process()
