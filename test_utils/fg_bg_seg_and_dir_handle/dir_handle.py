import os
import glob
import numpy as np
import scipy.misc
from PIL import Image
import re


def get_dirName(thedir):
    '''

    :param thedir:
    :return: 当前目录下所有非隐藏的子目录
    '''

    dirs = [name for name in os.listdir(thedir) if
              os.path.isdir(os.path.join(thedir, name)) and not name.startswith('.')]
    return dirs

def test_get_dirName():

    dir = "/p300/tiny_zh_shanghaitech/JPEGImages"
    dir = get_dirName(dir)
    print("test_get_dirName: ", dir)

    pass

def get_files(src_rootpath):

    dirs = sorted(glob.glob(os.path.join(src_rootpath, '*')))  # 扫描所有非隐藏目录并返回pathls
    # print(dirs)
    all_imgs = []
    for dir in dirs:
        dirname = dir.split('/')[-1]
        # print("dirname: ", dirname)
        imgs = sorted(glob.glob(os.path.join(dir, '*')))
        # print("imgs: ", imgs)
        all_imgs.append(imgs)
    all_imgs = np.concatenate(all_imgs)
    # print("len of all_imgs: ", len(all_imgs))
    return all_imgs

def test_get_files():
    dir = "/p300/tiny_zh_shanghaitech/JPEGImages"
    imgs = get_files(dir)
    print("imgs: ", len(imgs), imgs[0], imgs[1], imgs[-2], imgs[-1])
    pass

def generate_imageset_txt(src_rootpath, file_path):
    imgs = get_files(src_rootpath)
    with open(file_path, 'a') as f:
        for img in imgs:
            f.write(img)
            f.write("\n")
    pass

def test_generate_imageset_txt():
    # 测试下 imgs
    src_rootpath = "/p300/tiny_zh_shanghaitech/JPEGImages"
    file_path = "/p300/tiny_zh_shanghaitech/ImageSets/val.txt"
    generate_imageset_txt(src_rootpath, file_path)

    # 测试下 annotation
    # (先把npy转为png再生成path)

    pass

def npy2png(src_rootpath, dest_rootpath, num_bits_fileName):
    '''
    input: img and mask(xxx,npy)
    :return: xxx.png with dtype {0,1} (img & mask)
    '''

    # 分析 xxx.npy
    # np.load("01_0014.npy").shape: (265, 480, 856)
    # np.load("05_0020.npy").shape: (649, 480, 856)
    # 所以，mask shape is [num_samples, 480, 856]
    # 分析 img
    # cv2.imread("033.jpg").shape: (480, 856, 3)
    # 所以，理论上，直接恢复 xxx.npy为png即可，just try:

    files = sorted(glob.glob(os.path.join(src_rootpath, '*.npy')))  # 扫描所有*.npy并返回path
    print(files)

    for file in files:
        filename = file.split('/')[-1].split('.')[0]
        print("dirname: ", filename)
        out_path = os.path.join(dest_rootpath, filename)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        image_array_batch = np.load(file)
        print("image_array_batch: ", image_array_batch.shape)
        print("image_array_batch[0,-1]: ", image_array_batch[0].shape, image_array_batch[-1].shape)
        batch_size = image_array_batch.shape[0]
        for idx in range(batch_size):
            file_path = os.path.join(out_path, "{}.png".format(str(idx).zfill(num_bits_fileName)))
            # scipy.misc.imsave('{}.png'.format(file_path), image_array_batch[idx])
            # print("image_array_batch[idx]: ", image_array_batch[idx].max())
            im = Image.fromarray(np.uint8(image_array_batch[idx] * 255)) # 有两点注意：1)-要做img show，pixel value必须变为0-255
            # 2)-255默认是int32，所以乘完之后结果是int32，要转成uint8，否则Pillow无法识别会报错
            im.save(file_path)
    pass

def test_npy2png():
    src_rootpath = "/p300/tiny_zh_shanghaitech/npy_tmp"
    dest_rootpath = "/p300/tiny_zh_shanghaitech/Annotations"
    npy2png(src_rootpath, dest_rootpath, 3)
    pass

def generate_final_txt(img_rootpath, annotation_rootpath, out_path):
    '''

    :return: 生成一个 [num_samples, 2]的tensor, dtype为str, 第一列是 img_path, 第二列是 annotation_apath
    '''

    imgs = get_files(img_rootpath)
    annotations = get_files(annotation_rootpath)
    batch_size = len(imgs)
    with open(out_path, 'a') as f:
        for idx in range(batch_size):
            # 下面这个字符串截断的方法非常野蛮，仅本代码能work !!!
            loc = imgs[idx].find("JPEGImages")
            img_relative_path = imgs[idx][(loc-1):]
            loc = annotations[idx].find("Annotations")
            annotation_relative_path = annotations[idx][(loc-1):]
            f.write(img_relative_path+" "+annotation_relative_path)
            f.write("\n")
    pass

def test_generate_val_txt():

    '''
    input: imgs and annotation root_path
    :return: 生成一个 [num_samples, 2]的tensor, dtype为str, 第一列是 img_path, 第二列是 annotation_apath 的 txt file
    '''

    # 测试下 imgs
    img_rootpath = "/p300/tiny_zh_shanghaitech/JPEGImages"
    annotation_rootpath = "/p300/tiny_zh_shanghaitech/Annotations"
    out_path = "/p300/tiny_zh_shanghaitech/ImageSets/val.txt"
    generate_final_txt(img_rootpath, annotation_rootpath, out_path)

    pass

def avenue_npy2png():
    src_rootpath = "/p300/avenue/pixel_masks"
    dest_rootpath = "/p300/avenue/Annotations"
    npy2png(src_rootpath, dest_rootpath, 5)
    pass

def avenue_generate_val_txt():

    '''
    input: imgs and annotation root_path
    :return: 生成一个 [num_samples, 2]的tensor, dtype为str, 第一列是 img_path, 第二列是 annotation_apath 的 txt file
    '''

    img_rootpath = "/p300/avenue/JPEGImages"
    annotation_rootpath = "/p300/avenue/Annotations"
    out_path = "/p300/avenue/ImageSets/val.txt"
    generate_final_txt(img_rootpath, annotation_rootpath, out_path)

    pass

if __name__ == '__main__':
    # test_get_dirName()
    # test_get_files()
    # test_generate_imageset_txt()
    # test_npy2png()
    #
    # test_generate_val_txt()
    #
    # avenue_npy2png()
    avenue_generate_val_txt()


    pass

# 注意一个问题：因为上述txt file的写入方式都是a(追加)，所以确保要删除旧文件(比如
    # 我在向val.txt写入时就犯错了：一直往旧的val.txt追加新内容，我的本意是往空的新的val.txt添加新内容)