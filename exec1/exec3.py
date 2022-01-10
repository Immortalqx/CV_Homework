from scipy import io
import scipy
import imageio
import os

# TODO
# 预处理，但实际上给的数据已经是处理好了的！

# 读取mat文件中所有数据
# mat文件里面是以字典形式存储的数据
# 包括 dict_keys(['__globals__', 'groundTruth', '__header__', '__version__'])
# 我们要用到'groundTruth']中的轮廓
# x['groundTruth'][0][0][0][0][1]为轮廓
# x['groundTruth'][0][0][0][0][0]为分割图


def bsds_trans(root, num_anno):
    PATH = os.path.join(root, 'groundTruth')
    for sub_dir_name in ['train', 'test', 'val']:
        sub_pth = os.path.join(PATH, sub_dir_name)
        ##为生成的图片新建个文件夹保存
        save_pth = os.path.join(root, 'data/GT_convert_{}'.format(num_anno), sub_dir_name)
        os.makedirs(save_pth, exist_ok=True)

        print('开始转换' + sub_dir_name + '文件夹中内容')
        for index in range(len(os.listdir(sub_pth))):
            filename = os.listdir(sub_pth)[index]
            data = io.loadmat(os.path.join(sub_pth, filename))
            try:
                if len(data['groundTruth'][0]) < num_anno + 1: raise IndexError
                edge_data = data['groundTruth'][0][num_anno][0][0][1]
                edge_data_255 = edge_data * 255
                new_img_name = filename.split('.')[0] + '.jpg'
                print(new_img_name)
                imageio.imsave(os.path.join(save_pth, new_img_name), edge_data_255)

            except IndexError:
                index = min(len(os.listdir(sub_pth)) - 1, index + 1)
                filename = os.listdir(sub_pth)[index]


if __name__ == '__main__':
    # 运行时需要改变root值为BSD500所在的相应根目录
    root = 'data_set/BSDS500'
    # 选取不同标注者标注的label,范围（0，5）
    num_anno = 5
    bsds_trans(root, num_anno)
