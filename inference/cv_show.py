import os
import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt


def parse_txt(path):
    f = open(path)
    data = f.readlines()
    scores, boxes = [], []
    for ann in data[:-1]:
        ann = ann.split(' ')
        score = float(ann[0])
        xmin = int(ann[1])
        ymin = int(ann[2])
        xmax = int(ann[3])
        ymax = int(ann[4])
        scores.append(score)
        boxes.append([xmin, ymin, xmax, ymax])
    return scores, boxes

if __name__ == '__main__':
    txtPath = './output/lj_test2.txt'
    scores, boxes = parse_txt(txtPath)
    img = cv.imread('./images/lj_test2.tif')
    font = cv.FONT_HERSHEY_SIMPLEX  # 定义字体
    raw_h, raw_w = img.shape[:2]
    print(raw_h, raw_w)
    for i, (score, box) in enumerate(zip(scores, boxes)):
        print(i, score, box)
        if box[0] > raw_w or box[1] > raw_h:
            continue
        if box[2] < 0 or box[3] < 0:
            continue
        # img_ = img.copy()
        # bbox_left, bbox_top, bbox_right, bbox_bottom = int(box[0]), int(box[3]), int(box[2]), int(box[1])
        # crop_img = img_[bbox_bottom:bbox_top, bbox_left:bbox_right, :]
        cv.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        text_location = (int(box[0]) + 2, int(box[1]) - 4)
        img = cv.putText(img, f'{score * 100:.2f}%', text_location, font,
                            fontScale=0.5, color=(0, 0, 255))
        # plt.imshow(crop_img)
        # plt.show()
    cv.imwrite('lj_test2.tif', img)
