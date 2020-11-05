import argparse
import os
import platform
import matplotlib.pyplot as plt
import shutil
import time
from pathlib import Path
import numpy as np
import glob
import tqdm
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from torch.autograd import Variable

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.imagexy2shp import *


def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score


def detect():
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    gap = opt.gap
    # Initialize
    set_logging()
    device = select_device(opt.device)
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    # os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # imglist = glob.glob(f'{source}/*.tif')
    imglist = [source]
    font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
    font_size = 1
    frame_size = imgsz - gap
    t0 = time.time()

    # get class_dict from txt
    f = open('./data/cls_dict.txt')
    data = f.readlines()
    cls_dict = []
    for cls in data:
        cls = cls.replace('\n', '').strip()
        if cls == '':
            break
        cls_dict.append(cls)

    for j, imgPath in tqdm.tqdm(enumerate(imglist)):
        image_name = os.path.split(imgPath)[-1].split('.')[0]
        image = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        dataset = gdal.Open(imgPath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_h, raw_w = image.shape[:2]
        row = raw_h // frame_size + 1
        col = raw_w // frame_size + 1
        radius_h = row * frame_size - raw_h
        radius_w = col * frame_size - raw_w
        image = cv2.copyMakeBorder(image, 0, radius_h, 0, radius_w, cv2.BORDER_REFLECT)
        image = cv2.copyMakeBorder(image, 0, gap, 0, gap, cv2.BORDER_REFLECT)
        boxes, scores = [], []
        for i in tqdm.tqdm(range(row)):
            for j in range(col):
                image1 = image.copy()
                subImg = image1[i * frame_size:(i + 1) * frame_size + gap,
                         j * frame_size:(j + 1) * frame_size + gap, :]
                subImg_ = subImg.copy()
                subImg = subImg.astype(np.float32)
                subImg /= 255.0  # 0 - 255 to 0.0 - 1.0
                subImg = np.transpose(subImg, (2, 0, 1))
                subImg = Variable(torch.from_numpy(np.array([subImg])).cuda())
                subImg = subImg.half() if half else subImg.float()

                # Inference
                t1 = time_synchronized()
                pred = model(subImg, augment=opt.augment)[0]

                # Apply NMS
                preds = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                           agnostic=opt.agnostic_nms)
                try:
                    for pred in preds[0]:
                        pred = pred.cpu().numpy()
                        pred[:4] = pred[:4].astype(np.int32).clip(min=0, max=imgsz - 1)
                        pred[0] = pred[0] + j * frame_size
                        pred[1] = pred[1] + i * frame_size
                        pred[2] = pred[2] + j * frame_size
                        pred[3] = pred[3] + i * frame_size
                        boxes.append([pred[0], pred[1], pred[2], pred[3], pred[5].astype(np.int32)])
                        scores.append(pred[4])
                        # cv2.rectangle(subImg_, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 0, 255), 3)
                        # text_location = (int(pred[0]) + 2, int(pred[1]) - 4)
                        # subImg_ = cv2.putText(subImg_, f'garbage {pred[4] * 100:.2f}%', text_location, font,
                        #                      fontScale=0.5, color=(0, 0, 255))
                    # plt.imshow(subImg_)
                    # plt.show()
                except:
                    continue

        # 丢弃原图像边界外的框
        boxes, scores = np.array(boxes), np.array(scores)
        keep = (boxes[:, 0] < raw_w) & (boxes[:, 1] < raw_h)
        boxes = boxes[keep]
        scores = scores[keep]
        assert len(boxes) == len(scores), print(f'length of boxes :{len(boxes)}, length of scores :{len(scores)}')
        boxes, scores = nms(boxes, scores, opt.iou_thres)
        boxes_, scores_, clss_ = [], [], []
        for box, score in zip(boxes, scores):
            # with open(f"{out}/{image_name}.txt", 'a+') as f:
            #     f.write(f"{score} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])} \n")
            xmin, ymin = imagexy2geo(dataset, int(box[0]), int(box[1]))
            xmax, ymax = imagexy2geo(dataset, int(box[2]), int(box[3]))
            boxes_.append([xmin, ymin, xmax, ymax])
            scores_.append(int(score * 100))
            clss_.append(box[4])
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            text_location = (int(box[0]) + 2, int(box[1]) - 4)
            image = cv2.putText(image, f'{score * 100:.2f}%', text_location, font,
                                 fontScale=0.5, color=(0, 0, 255))
        # plt.imshow(image)
        # plt.show()
        # cv2.imwrite(os.path.join(out, f'{image_name}.tif'), image)

        # results2shp
        # imagexy2shp(imgPath, f"{out}/{image_name}.shp", boxes_, scores_)
        imagexy2shp(imgPath, out, boxes_, scores_, clss_, cls_dict)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    f = open('config_test.txt')
    data = f.readlines()
    source = data[0].replace('\n', '')
    weigths = data[1].replace('\n', '')
    output = data[2].replace('\n', '')
    out_root = os.path.split(output)[0]
    assert os.path.exists(out_root), print(f'The out path is not existed, please create this path:{out_root}')

    # print(source, weigths, output)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=weigths, help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=source, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default=output, help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--gap', type=int, default=100, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
