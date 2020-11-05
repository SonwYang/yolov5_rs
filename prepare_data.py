import os
import cv2 as cv
import glob
import numpy as np
import shutil
import random
from random import sample
from shutil import copyfile
from utils.shp2imagexy import *
from utils.xml2yolo import *
import yaml


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def read_img(path):
    data = gdal.Open(path)
    lastChannel = data.RasterCount + 1
    arr = [data.GetRasterBand(idx).ReadAsArray() for idx in range(1, 4)]
    arr = np.dstack(arr)
    return arr


def prepare_data(imgRoots):
    if os.path.exists('./data/labels'):
        shutil.rmtree('./data/labels')
    if os.path.exists('./data/images'):
        shutil.rmtree('./data/images')
    if os.path.exists('./data/train'):
        shutil.rmtree('./data/train')
    if os.path.exists('./data/val'):
        shutil.rmtree('./data/val')
    if os.path.exists('./data/cls_dict.txt'):
        os.remove('./data/cls_dict.txt')

    imglist = []
    for imgRoot in imgRoots:
        subPaths = glob.glob(f'{imgRoot}/*.tif')
        for subPath in subPaths:
            imglist.append(subPath)

    # imglist = glob.glob(f'{imgRoot}/*/*.tif')
    mkdir('./data/labels')
    mkdir('./data/images')
    cls_dict = []
    for imgPath in imglist:
        imgName = os.path.split(imgPath)[-1].split('.')[0]
        subRoot = os.path.split(imgPath)[0]
        img = read_img(imgPath)
        w, h, c = img.shape
        shpPath = glob.glob(f'{subRoot}/*.shp')[0]
        # shpPath = imgPath.replace('tif', 'shp')
        anns = shp2imagexy(imgPath, shpPath)
        for ann in anns:
            cls = str(ann[4])
            if cls not in cls_dict:
                cls_dict.append(cls)
            x, y, w_, h_ = convert((w, h), ann[:-1])
            with open(f"./data/labels/{imgName}.txt", 'a+') as f:
                f.write(f"{cls_dict.index(cls)} {x} {y} {w_} {h_} \n")
        cv.imwrite(f'./data/images/{imgName}.jpg', img)

    for cls in cls_dict:
        with open("./data/cls_dict.txt", 'a+') as f:
            f.write(f"{cls} \n")

    split_trainval()

    # dataMap:
    dataMap = {
                'train': './data/train/images',
                'val': './data/val/images',
                'nc': len(cls_dict),
                'names': '.',
              }
    if os.path.exists('./data/data.yaml'):
        os.remove('./data/data.yaml')
    f = open('./data/data.yaml', "w", encoding='utf-8')
    yaml.dump(dataMap, f)
    f.close()
    return cls_dict

def split_trainval():
    if os.path.exists('./data/train'):
        shutil.rmtree('./data/train')
    if os.path.exists('./data/val'):
        shutil.rmtree('./data/val')
    train_root = './data/train'
    train_images_path = './data/train/images'
    train_labels_path = './data/train/labels'
    val_root = './data/val'
    val_images_path = './data/val/images'
    val_labels_path = './data/val/labels'
    mkdir(train_root)
    mkdir(train_images_path)
    mkdir(train_labels_path)
    mkdir(val_root)
    mkdir(val_images_path)
    mkdir(val_labels_path)
    txtlist = os.listdir('./data/labels')
    txtlist = [txtpath[:-4] for txtpath in txtlist]
    assert len(txtlist) > 0, print('The number of labels is empty')

    random.shuffle(txtlist)
    indices = list(range(len(txtlist)))
    indices = sample(indices, len(indices))
    split = int(np.floor(0.25 * len(txtlist)))
    train_idx, valid_idx = indices[split:], indices[:split]
    for idx in train_idx:
        fileName = txtlist[idx]
        copyfile(f'./data/labels/{fileName}.txt', f'{train_labels_path}/{fileName}.txt')
        copyfile(f'./data/images/{fileName}.jpg', f'{train_images_path}/{fileName}.jpg')

    for idx in valid_idx:
        fileName = txtlist[idx]
        copyfile(f'./data/labels/{fileName}.txt', f'{val_labels_path}/{fileName}.txt')
        copyfile(f'./data/images/{fileName}.jpg', f'{val_images_path}/{fileName}.jpg')


if __name__ == '__main__':
    # imgRoot = 'J:/dl_dataset/test'
    f = open('E:/2_1/yolov5-master/config_train2.txt')
    data = f.readlines()
    dataRoots = data[0][:-1].split(';')
    prepare_data(dataRoots)
    split_trainval()





