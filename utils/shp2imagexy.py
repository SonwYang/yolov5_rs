# -*- coding: utf-8 -*-
from osgeo import ogr
from osgeo import gdal
from osgeo import osr
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math


def getSRSPair(dataset):
    '''
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs

def xy_to_coor(x, y):
    lonlat_coordinate = []
    L = 6381372 * math.pi*2
    W = L
    H = L/2
    mill = 2.3
    lat = ((H/2-y)*2*mill)/(1.25*H)
    lat = ((math.atan(math.exp(lat))-0.25*math.pi)*180)/(0.4*math.pi)
    lon = (x-W/2)*360/W
    # TODO 最终需要确认经纬度保留小数点后几位
    lonlat_coordinate.append((round(lon,7),round(lat,7)))
    return round(lon,7), round(lat,7)


def geo2lonlat(dataset, x, y):
    '''
    将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param x: 投影坐标x
    :param y: 投影坐标y
    :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(x, y)
    return coords[:2]


def lonlat2geo(dataset, lon, lat):
    '''
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]


def imagexy2geo(dataset, col, row):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
    trans = dataset.GetGeoTransform()
    print(trans)
    print(row,col)
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]
    return px, py


def geo2imagexy01(dataset, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    # trans = dataset
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)

def geo2imagexy(dataset, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    #a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    #b = np.array([x - trans[0], y - trans[3]])
    #return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解

    dTemp = trans[1] * trans[5] - trans[2] * trans[4]
    Xpixel = (trans[5] * (x - trans[0]) - trans[2] * (y - trans[3])) / dTemp
    Yline = (trans[1] * (y - trans[3]) - trans[4] * (x - trans[0])) / dTemp
    return [Xpixel, Yline]


def shp2imagexy(imgPath, shpPath):
    dataset = gdal.Open(imgPath)
    ds = ogr.Open(shpPath,1)
    if ds is None:
        print('Could not open folder')
    in_lyr = ds.GetLayer()

    lyr_dn = in_lyr.GetLayerDefn()
    cls_index = lyr_dn.GetFieldIndex("Id")
    cls_name_g = lyr_dn.GetFieldDefn(cls_index)
    feature = in_lyr.GetNextFeature()

    fieldName = cls_name_g.GetNameRef()

    finalResult = []
    while feature is not None:
        geom = feature.geometry()
        id = feature.GetField('cls')
        arr = np.array(feature.GetGeometryRef().GetEnvelope())
        # print('before', arr)
        # coordsMin = lonlat2geo(dataset, arr[0], arr[3])
        coordsMin = geo2imagexy(dataset, arr[0], arr[3])
        # coordsMax = lonlat2geo(dataset, arr[1], arr[2])
        coordsMax = geo2imagexy(dataset, arr[1], arr[2])
        finalResult.append([coordsMin[0], coordsMax[0], coordsMin[1], coordsMax[1], id])

        feature = in_lyr.GetNextFeature()

    return finalResult


if __name__ == '__main__':
    img_filename = './test/0000000004_image.tif'
    dst_filename = './test/label3.shp'
    finalResult = shp2imagexy(img_filename, dst_filename)
    img = cv.imread(img_filename, cv.IMREAD_LOAD_GDAL)
    finalResult = np.array(finalResult)
    for bbox in finalResult:
        # xmin = min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
        # ymin = min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
        # xmax = max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
        # ymax = max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
        # cv.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 100, 255), 5)
        print(bbox)
        cv.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 100, 255), 5)

    plt.imshow(img)
    plt.show()