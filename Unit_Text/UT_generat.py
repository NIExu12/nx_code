# -*- coding: utf-8 -*-
import operator
import cv2
import os
import re
import random
import json
import copy
import numpy as np
from pathlib import Path



# 画塔顶
def draw_tading(image, center, tading_radius, img_w, img_h):

    # 颜色（红色，BGR格式）
    color = (0, 255, 255)

    # 画圆
    cv2.circle(image, center, tading_radius, color, thickness=-1)  # thickness=-1 表示实心圆

    cls = 'td'
    x_min = center[0] - tading_radius
    y_min = center[1] - tading_radius
    x_max = center[0] + tading_radius
    y_max = center[1] + tading_radius
    if ((y_min <= 0 and y_max <= 0) or
            (x_min <= 0 and x_max <= 0) or
            (y_min >= img_h and y_max >= img_h) or
            (x_min >= img_w and x_max >= img_w)):
        tading_coordinate_list = None
    else:
        tading_coordinate_list = [cls, (x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
    return tading_coordinate_list


# 画横担
def draw_hengdan(image, hengdan_w, hengdan_h, center, angle_degrees, img_w, img_h):
    # 颜色（绿色，BGR格式）
    color = (0, 255, 0)

    center_x = center[0]
    center_y = center[1]

    crossbar_vertices = [
        (int(center_x - hengdan_h * np.cos(angle_degrees) + hengdan_w * np.sin(angle_degrees)),
         int(center_y - hengdan_h * np.sin(angle_degrees) - hengdan_w * np.cos(angle_degrees))),
        (int(center_x + hengdan_h * np.cos(angle_degrees) + hengdan_w * np.sin(angle_degrees)),
         int(center_y + hengdan_h * np.sin(angle_degrees) - hengdan_w * np.cos(angle_degrees))),
        (int(center_x + hengdan_h * np.cos(angle_degrees) - hengdan_w * np.sin(angle_degrees)),
         int(center_y + hengdan_h * np.sin(angle_degrees) + hengdan_w * np.cos(angle_degrees))),
        (int(center_x - hengdan_h * np.cos(angle_degrees) - hengdan_w * np.sin(angle_degrees)),
         int(center_y - hengdan_h * np.sin(angle_degrees) + hengdan_w * np.cos(angle_degrees)))
    ]

    # 绘制横担
    cv2.fillPoly(image, [np.array(crossbar_vertices)], color)

    crossbar_vertices_x = []
    crossbar_vertices_y = []
    for i in crossbar_vertices:
        crossbar_vertices_x.append(int(i[0]))
        crossbar_vertices_y.append(int(i[1]))

    # 长方形外接矩形框左上角和右下角的坐标
    top_left = (min(crossbar_vertices_x), min(crossbar_vertices_y))
    bottom_right = (max(crossbar_vertices_x), max(crossbar_vertices_y))
    cls = 'hd'

    if ((top_left[1] <= 0 and bottom_right[1] <= 0) or
            (top_left[0] <= 0 and bottom_right[0] <= 0) or
            (top_left[1] >= img_h and bottom_right[1] >= img_h) or
            (top_left[0] >= img_w and bottom_right[0] >= img_w)):
        tading_coordinate_list = None
        jueyuanzi_crossbars = None
    else:
        jueyuanzi_crossbars = []
        tading_coordinate_list = []
        tading_coordinate_list.append(cls)
        for crossbar in crossbar_vertices:
            jueyuanzi_crossbars.append(crossbar)
            tading_coordinate_list.append(crossbar)
        jueyuanzi_crossbars.append((int((crossbar_vertices[0][0] + crossbar_vertices[3][0]) / 2),
                                    int((crossbar_vertices[0][1] + crossbar_vertices[3][1]) / 2)))
        jueyuanzi_crossbars.append((int((crossbar_vertices[1][0] + crossbar_vertices[2][0]) / 2),
                                    int((crossbar_vertices[1][1] + crossbar_vertices[2][1]) / 2)))

    return tading_coordinate_list, jueyuanzi_crossbars


# 画绝缘子
def draw_jueyuanzi(image, center_list, jueyuanzi_radius, img_w, img_h):

    # 颜色（红色，BGR格式）
    color = (0, 0, 205)
    jueyuanzi_labels_list = []

    if center_list is not None:
        for center in center_list:
            cv2.circle(image, center, jueyuanzi_radius, color, thickness=-1)  # thickness=-1 表示实心圆

            cls = 'jyz_tc_bs'
            x_min = center[0] - jueyuanzi_radius
            y_min = center[1] - jueyuanzi_radius
            x_max = center[0] + jueyuanzi_radius
            y_max = center[1] + jueyuanzi_radius
            if ((y_min <= 0 and y_max <= 0) or
                    (x_min <= 0 and x_max <= 0) or
                    (y_min >= img_h and y_max >= img_h) or
                    (x_min >= img_w and x_max >= img_w)):
                jueyuanzi_labels = None
            else:
                jueyuanzi_labels = [cls, (x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
            jueyuanzi_labels_list.append(jueyuanzi_labels)
        jueyuanzi_labels_list = list(filter(None, jueyuanzi_labels_list))
        return jueyuanzi_labels_list


def on_segment(p, q, r):
    """
    检查点 q 是否在由 p 和 r 构成的线段上
    """
    return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])


def orientation(p, q, r):
    """
    计算三个点的方向
    返回值：
     -1: 逆时针方向
      0: 共线
      1: 顺时针方向
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else -1


def do_intersect(p1, q1, p2, q2):
    """
    检查线段 p1q1 和 p2q2 是否相交
    """
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # 一般情况
    if o1 != o2 and o3 != o4:
        return True

    # 特殊情况：p1, q1, p2 共线，且 p2 在 p1q1 上
    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    # 特殊情况：p1, q1, q2 共线，且 q2 在 p1q1 上
    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    # 特殊情况：p2, q2, p1 共线，且 p1 在 p2q2 上
    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    # 特殊情况：p2, q2, q1 共线，且 q1 在 p2q2 上
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


# 计算两直线之间的交点
def cross_point_func(line1, line2):  # 计算交点函数
    x1 = line1[0][0]  # 取直线1的第一个点坐标
    y1 = line1[0][1]
    x2 = line1[1][0]  # 取直线1的第二个点坐标
    y2 = line1[1][1]

    x3 = line2[0][0]  # 取直线2的第一个点坐标
    y3 = line2[0][1]
    x4 = line2[1][0]  # 取直线2的第二个点坐标
    y4 = line2[1][1]

    # 线段1的两个端点
    p1 = np.array([x1, y1])
    q1 = np.array([x2, y2])

    # 线段2的两个端点
    p2 = np.array([x3, y3])
    q2 = np.array([x4, y4])

    # 检查线段是否相交
    if do_intersect(p1, q1, p2, q2):
        # 检查线性方程组的系数矩阵是否为奇异矩阵
        if not (np.array_equal(p1, q1) or np.array_equal(p2, q2)):
            # 构建线性方程组的系数矩阵
            coefficients = np.array([[-(y2 - y1), x2 - x1], [-(y4 - y3), x4 - x3]])

            # 构建等式右侧的常数向量
            constants = np.array([y1 * (x2 - x1) - x1 * (y2 - y1), y3 * (x4 - x3) - x3 * (y4 - y3)])

            try:
                # 解线性方程组
                intersection = np.linalg.solve(coefficients, constants)

                # 检查交点是否在两条线段上
                if on_segment(p1, intersection, q1) and on_segment(p2, intersection, q2):
                    return tuple(intersection)
            except np.linalg.LinAlgError:
                pass

    # 无交点
    return None

def add_cls(labels, json_path):
    try:
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print(f"文件不存在")
        data = {"detections": []}
    if len(labels[0]) > 2:
        for label in labels:
            new_cls = {
                "class_name": label[0],
                "score": 1,
                "coordinates": [
                    label[1],
                    label[2],
                    label[3],
                    label[4]
                ]
            }
            data["detections"].append(new_cls)
    else:
        new_cls = {
            "class_name": labels[0],
            "score": 1,
            "coordinates": [
                labels[1],
                labels[2],
                labels[3],
                labels[4]
            ]
        }
        data["detections"].append(new_cls)

    # 将数据保存到JSON文件
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


# 输出生成json格式
def output_json(image_name,
                json_path,
                img_w, img_h,
                hengdan_labels,
                tading_labels,
                jueyuanzi_labels):
    data = {
        "image_name": image_name,
        "width": img_w,
        "height": img_h,
        "detections": [
        ]
    }

    # 将数据写入JSON文件
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)  # 使用indent参数添加缩进，使JSON更易读

    add_cls(hengdan_labels, json_path)
    add_cls(tading_labels, json_path)
    add_cls(jueyuanzi_labels, json_path)


def get_labels(coordinates, img_w, img_h):
    cls = coordinates[0]
    del coordinates[0]
    coordinates.sort(key=lambda x:(x[0], x[1]))
    # coordinates.sort(key=operator.itemgetter(0))
    img_line1 = ((0, 0), (img_w, 0))
    img_line2 = ((0, 0), (0, img_h))
    img_line3 = ((img_w, img_h), (0, img_h))
    img_line4 = ((img_w, img_h), (img_w, 0))
    img_line_list = [img_line1, img_line2, img_line3, img_line4]

    coordinates_line1 = ((-100, 828), (295, 1034))
    coordinates_line2 = ((-100, 828), (586, -492))
    coordinates_line3 = ((982, -286), (586, -492))
    coordinates_line4 = ((982, -286), (295, 1034))

    # coordinates_line1 = (coordinates[0], coordinates[1])
    # coordinates_line2 = (coordinates[0], coordinates[2])
    # coordinates_line3 = (coordinates[3], coordinates[2])
    # coordinates_line4 = (coordinates[3], coordinates[1])
    coordinates_line_list = [coordinates_line1, coordinates_line2, coordinates_line3, coordinates_line4]
    cross_points = []

    for j in img_line_list:
        for k in coordinates_line_list:
            cross_points.append(cross_point_func(j, k))
    cross_points = list(filter(None, cross_points))

    new_coordinates = []
    for i, m in enumerate(coordinates):
        if 0 < m[0] < img_w and 0 < m[1] < img_h:
            new_coordinates.append(coordinates[i])

    for n in cross_points:
        new_coordinates.append(n)
    rectangular = []
    rectangular.append(cls)
    coordinates_list = External_Rectangular(new_coordinates)
    for co in coordinates_list:
        rectangular.append(co)
    return rectangular


def External_Rectangular(coordinates):
    x_list = []
    y_list = []
    for i in coordinates:
        if i is not None:
            x_list.append(i[0])
            y_list.append(i[1])
    rectangular = [int(abs(min(x_list))), int(abs(min(y_list))), int(abs(max(x_list))), int(abs(max(y_list)))]
    return rectangular

# 文件夹路径自动排序
def increment_path(path, exist_ok=False):
    path = Path(path)
    if exist_ok or not path.exists():
        return path
    root = path.stem
    suffix = path.suffix
    dirs = [d for d in path.parent.iterdir() if d.is_dir() and d.stem.startswith(root)]
    matches = [
        int(re.search(r"(?<=_)\d+", str(d.stem)).group())
        for d in dirs
        if re.search(r"(?<=_)\d+", str(d.stem))
    ]
    i = 2
    if matches:
        i = max(matches) + 1
    return path.with_name(f"{root}_{i}{suffix}")


# 图片路径自动排序
def increment_img_path(path, exist_ok=False):
    path = Path(path)
    if exist_ok or not path.exists():
        return path
    root = path.stem
    suffix = path.suffix
    dirs = [d for d in path.parent.iterdir() if d.is_dir() and d.stem.startswith(root)]
    matches = [
        int(re.search(r"(?<=_)\d+", str(d.stem)).group())
        for d in dirs
        if re.search(r"(?<=_)\d+", str(d.stem))
    ]
    i = 2
    if matches:
        i = max(matches) + 1
    while True:
        new_path = path.with_name(f"{root}_{i}{suffix}")
        if not new_path.exists():
            break
        i += 1
    return new_path


img_w = 1920    # 图片宽
img_h = 1080    # 图片高
run_amount = 100    # 生成数量
unit_text_path = increment_path("runs/unit_text", exist_ok=False)   # 存放路径

for i in range(run_amount):

    tading_radius = random.randint(0, int(img_h/8))
    jueyuanzi_radius = int(tading_radius * 0.8)
    center = (random.randint(-tading_radius,
                             img_w+tading_radius),
              random.randint(-tading_radius,
                             img_h+tading_radius))
    hengdan_w = tading_radius * 6
    hengdan_h = tading_radius * 1.8

    # 随机生成横担的角度
    angle_degrees = random.uniform(0, 360)

    # 创建一个黑色背景
    image = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    hengdan_coordinates, crossbar_vertices = draw_hengdan(image, hengdan_w, hengdan_h, center, angle_degrees, img_w, img_h)
    tading_coordinates = draw_tading(image, center, tading_radius, img_w, img_h)
    jueyuanzi_coordinates = draw_jueyuanzi(image, crossbar_vertices, jueyuanzi_radius, img_w, img_h)

    os.makedirs(unit_text_path, exist_ok=True)
    img_path = increment_img_path(unit_text_path / ('ut' + ".jpg"), exist_ok=False)
    json_path = increment_img_path(unit_text_path / ('ut' + ".json"), exist_ok=False)
    image_name = Path(img_path).name

    # 保存图片
    cv2.imwrite(str(img_path), image)

    if hengdan_coordinates is not None:
        hengdan_labels = get_labels(hengdan_coordinates, img_w, img_h)
        tading_labels = get_labels(tading_coordinates, img_w, img_h)
        jueyuanzi_labels = []
        for j in jueyuanzi_coordinates:
            jueyuanzi_labels.append(get_labels(j, img_w, img_h))
        # 生成json文件并保存
        output_json(image_name, json_path, img_w, img_h, hengdan_labels, tading_labels, jueyuanzi_labels)
