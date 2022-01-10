import cv2.cv2 as cv
from matplotlib import pyplot as plt


def MyBlur(image, title=None, sigma=0, display=True):
    images = [image]
    for ks in range(3, 30, 2):
        img = cv.GaussianBlur(image, (ks, ks), sigma)
        # cv.imshow("ksize=(%d,%d)" % (ks, ks), img)
        # cv.waitKey(0)
        images.append(img)

    if not display:
        return

    if title is None:
        fig = plt.figure("Blur", figsize=(10, 6))  # figure size in inches
    else:
        fig = plt.figure(title, figsize=(10, 6))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95, hspace=0.25, wspace=0.1)

    for i in range(len(images)):
        if i == 0:
            ax = fig.add_subplot(5, 5, i + 1, xticks=[], yticks=[], title="origin")
        else:
            ax = fig.add_subplot(5, 5, i + 1, xticks=[], yticks=[], title="ksize=(%d,%d)" % (i * 2 + 1, i * 2 + 1))
        img = cv.cvtColor(images[i], cv.COLOR_BGR2RGB)
        ax.imshow(img)


if __name__ == "__main__":
    print("调整 start 参数值和 end 参数值以显示更多尺度因子下的滤波结果！")
    start = 1
    end = 11
    src1 = cv.imread("data_set/lena512color.tiff")
    src2 = cv.imread("data_set/lena512.bmp")
    for sigma in range(1, 12):
        if sigma > end:
            break
        if sigma < start:
            continue
        MyBlur(src1, "Color_Blur(sigma=%d)" % sigma, sigma)
        MyBlur(src2, "Gray_Blur(sigma=%d)" % sigma, sigma)

    plt.show()
