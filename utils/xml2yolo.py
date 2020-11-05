import os
import numpy as np
import glob
import xml.etree.ElementTree as ET
import cv2 as cv


def parse_xml(path):
    tree = ET.parse(path)
    size = tree.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    root = tree.findall('object')
    boxes_list = []
    for sub in root:
        xmin = float(sub.find('bndbox').find('xmin').text)
        xmax = float(sub.find('bndbox').find('xmax').text)
        ymin = float(sub.find('bndbox').find('ymin').text)
        ymax = float(sub.find('bndbox').find('ymax').text)
        boxes_list.append([xmin, xmax, ymin, ymax])
    return np.array(boxes_list), w, h


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    xmlList = glob.glob(f'./Annotations/*.xml')
    out_root = 'label'
    mkdir(out_root)
    for xmlPath in xmlList:
        imgPath = xmlPath[:-3].replace('Annotations', 'JPEGImages') + 'jpg'
        imgid = os.path.split(imgPath)[-1].split('.')[0]
        img = cv.imread(imgPath)
        shape = img.shape
        bboxes, w, h = parse_xml(xmlPath)
        for b in bboxes:
            b_ = convert((w, h), b)
            with open(f"{out_root}/{imgid}.txt", 'a+') as f:
                f.write(f"0 {float(b_[0])} {float(b_[1])} {float(b_[2])} {float(b_[3])} \n")
