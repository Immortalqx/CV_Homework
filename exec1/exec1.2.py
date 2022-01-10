import cv2.cv2 as cv


# 一开始写的canny，发现不符合要求，就放到这个函数里面了
def my_canny():
    src = cv.imread("data_set/lena512.bmp")
    src = cv.GaussianBlur(src, (3, 3), 0)
    src = cv.Canny(src, 75, 150)
    cv.imshow("Canny", src)
    cv.waitKey(0)


if __name__ == "__main__":
    # 读取图片
    image_gray = cv.imread("data_set/lena512.bmp", 0)
    image_color = cv.imread("data_set/lena512color.tiff")
    # 高斯滤波
    smooth_image_gray = cv.GaussianBlur(image_gray, (5, 5), 0)
    smooth_image_color = cv.GaussianBlur(image_color, (5, 5), 0)
    # 拉普拉斯变换
    gray_lap_gray = cv.Laplacian(smooth_image_gray, cv.CV_16S, ksize=3)
    gray_lap_color = cv.Laplacian(smooth_image_color, cv.CV_16SC3, ksize=3)
    # 绝对值处理
    laplacian_result_gray = cv.convertScaleAbs(gray_lap_gray)
    laplacian_result_color = cv.convertScaleAbs(gray_lap_color)
    # 阈值化处理
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    final_gray = cv.morphologyEx(laplacian_result_gray, cv.MORPH_TOPHAT, kernel)
    final_color = cv.morphologyEx(laplacian_result_color, cv.MORPH_TOPHAT, kernel)
    # final = laplacian_result.copy()
    _, final_gray = cv.threshold(final_gray, 125, 255, cv.THRESH_TOZERO_INV)
    _, final_gray = cv.threshold(final_gray, 30, 255, cv.THRESH_BINARY)
    _, final_color = cv.threshold(final_color, 125, 255, cv.THRESH_TOZERO_INV)
    _, final_color = cv.threshold(final_color, 30, 255, cv.THRESH_BINARY)
    # 显示最后的结果
    cv.imshow("laplacian gray", laplacian_result_gray)
    cv.imshow("laplacian color", laplacian_result_color)
    cv.imshow("final gray", final_gray)
    cv.imshow("final color", final_color)

    # # 与Canny做对比
    # dst = cv.Canny(smooth_image, 75, 150)
    # cv.imshow("Canny", dst)

    cv.waitKey(0)
