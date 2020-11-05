# -*- coding: utf-8 -*-
import gdal
from osgeo import ogr, osr


def read_img(filename):
    dataset=gdal.Open(filename)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)

    # del dataset
    return im_width,im_height,im_proj,im_geotrans,im_data,dataset


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


def imagexy2geo(dataset, col, row):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
    trans = dataset.GetGeoTransform()
    # print(trans)
    # print(row,col)
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]
    return px, py


def imagexy2shp(img_path, strVectorFile, bboxes, scores, clss, cls_dict):
    im_width,im_height,im_proj,im_geotrans,im_data,dataset = read_img(img_path)
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")  # 为了支持中文路径
    gdal.SetConfigOption("SHAPE_ENCODING", "CP936")  # 为了使属性表字段支持中文
    ogr.RegisterAll()
    strDriverName = "ESRI Shapefile"  # 创建数据，这里创建ESRI的shp文件
    oDriver = ogr.GetDriverByName(strDriverName)
    if oDriver == None:
        print("%s 驱动不可用！\n", strDriverName)

    oDS = oDriver.CreateDataSource(strVectorFile)  # 创建数据源
    if oDS == None:
        print("创建文件【%s】失败！", strVectorFile)

    # srs = osr.SpatialReference()  # 创建空间参考
    # srs.ImportFromEPSG(4326)  # 定义地理坐标系WGS1984
    srs = osr.SpatialReference(wkt=dataset.GetProjection())#我在读栅格图的时候增加了输出dataset，这里就可以不用指定投影，实现全自动了，上面两行可以注释了，并且那个proj参数也可以去掉了，你们自己去掉吧
    papszLCO = []
    # 创建图层，创建一个多边形图层,"TestPolygon"->属性表名
    oLayer = oDS.CreateLayer("TestPolygon", srs, ogr.wkbPolygon, papszLCO)
    if oLayer == None:
        print("图层创建失败！\n")

    '''下面添加矢量数据，属性表数据、矢量数据坐标'''
    oFieldID = ogr.FieldDefn("cls", ogr.OFTString)  # 创建一个叫FieldID的整型属性
    oLayer.CreateField(oFieldID, 1)

    oFieldID = ogr.FieldDefn("Confidence", ogr.OFTInteger)  # 创建一个叫FieldID的整型属性
    oLayer.CreateField(oFieldID, 1)
    # oFieldName = ogr.FieldDefn("FieldName", ogr.OFTString)  # 创建一个叫FieldName的字符型属性
    # oFieldName.SetWidth(100)  # 定义字符长度为100
    # oLayer.CreateField(oFieldName, 1)

    oDefn = oLayer.GetLayerDefn()  # 定义要素
    # 创建单个面
    for bbox, score, cls in zip(bboxes, scores, clss):
        oFeatureTriangle = ogr.Feature(oDefn)
        oFeatureTriangle.SetField(0, cls_dict[int(cls)])  # 第一个参数表示第几个字段，第二个参数表示字段的值
        oFeatureTriangle.SetField(1, score)
        # oFeatureTriangle.SetField(1, "单个面")
        ring = ogr.Geometry(ogr.wkbLinearRing)  # 构建几何类型:线
        ring.AddPoint(bbox[0], bbox[1])  # 添加点01
        ring.AddPoint(bbox[2], bbox[1])  # 添加点02
        ring.AddPoint(bbox[2], bbox[3])  # 添加点03
        ring.AddPoint(bbox[0], bbox[3])  # 添加点04
        yard = ogr.Geometry(ogr.wkbPolygon)  # 构建几何类型:多边形
        yard.AddGeometry(ring)
        yard.CloseRings()

        geomTriangle = ogr.CreateGeometryFromWkt(str(yard))  # 将封闭后的多边形集添加到属性表
        oFeatureTriangle.SetGeometry(geomTriangle)
        oLayer.CreateFeature(oFeatureTriangle)

    oDS.Destroy()
    print("数据集创建完成！\n")


def parse_txt(path):
    f = open(path)
    data = f.readlines()
    anns = []
    for ann in data:
        try:
            ann = ann.split(' ')
            xmin = float(ann[0])
            ymin = float(ann[1])
            xmax = float(ann[2])
            ymax = float(ann[3])
            anns.append([xmin, ymin, xmax, ymax])
        except:
            print('error', ann)
    return anns


if __name__ == "__main__":
    img_filename = './test/0000000004_image.tif'
    dst_filename = './test/label6.shp'
    # imagexy2shp(img_filename, dst_filename)
    txtPath = './test/0000000004_image.txt'
    dataset = gdal.Open(img_filename)
    anns = parse_txt(txtPath)
    annsGEO = []
    for ann in anns:
        xmin, ymin = imagexy2geo(dataset, ann[0], ann[1])
        xmax, ymax = imagexy2geo(dataset, ann[2], ann[3])
        annsGEO.append([xmin, ymin, xmax, ymax])

    # 转换成面矢量
    imagexy2shp(img_filename, dst_filename, annsGEO)
