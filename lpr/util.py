import numpy as np
import cv2 as cv

# 通过面积和长宽比来判断车牌是否有效


def verify_plate_sizes(contour):
    (center_x, center_y), (w, h), angle = cv.minAreaRect(contour)
    w = int(w)
    h = int(h)
    MIN_ASPECT_RATIO = 2.0		    # 车牌区域轮廓矩形的最小长宽比
    MAX_ASPECT_RATIO = 5.0		    # 车牌区域轮廓矩形的最大长宽比 /tried 4
    MIN_AREA = 34.0 * 8.0 * 10      # 车牌面积的最小值
    MAX_AREA = 34.0 * 8.0 * 110     # 车牌面积的最大值 /tried 34.0 * 8.0 * 100

    # 检查面积
    area = w * h
    if area > MAX_AREA or area < MIN_AREA:
        return False

    aspect = w / h
    if aspect < 1:
        aspect = 1 / aspect

    # 检查长宽比
    if aspect > MAX_ASPECT_RATIO or aspect < MIN_ASPECT_RATIO:
        return False

    return True

# 通过旋转矫正倾斜的车牌区域


def rotate_plate_image(contour, plate_image):
    # 获取该等值线框对应的外接正交矩形(长和宽分别与水平和竖直方向平行)
    x, y, w, h = cv.boundingRect(contour)
    top_most = tuple(contour[contour[:, :, 1].argmin()][0])
    bounding_image = plate_image[y: y + h, x: x + w]

    # rect结构： (center (x,y), (width, height), angle of rotation)
    rect = cv.minAreaRect(contour)
    rect_width, rect_height = np.int0(rect[1])      # 转成整数
    angle = np.abs(rect[2])             # 获得畸变角度

    # 如果宽度比高度小，说明矩形相对于最低角点而言，在第二象限；否则相对于最低角点而言在第一象限
    if rect_width < rect_height:
        temp = rect_height
        rect_height = rect_width
        rect_width = temp
        angle = 90 + angle

    if angle <= 5.0 or angle >= 175.0:                    # 对于较小的畸变角度，不予处理
        return bounding_image

    # 创建一个放大的图像，以便存放之前图像旋转后的结果
    enlarged_width = w * 3 // 2
    enlarged_height = h * 3 // 2
    enlarged_image = np.zeros(
        (enlarged_height, enlarged_width, plate_image.shape[2]), dtype=plate_image.dtype)
    x_in_enlarged = (enlarged_width - w) // 2
    y_in_enlarged = (enlarged_height - h) // 2
    roi_image = enlarged_image[y_in_enlarged:y_in_enlarged +
                               h, x_in_enlarged:x_in_enlarged+w]
    # 将旋转前的图像拷贝到放大图像的中心位置，注意，为了图像完整性，应拷贝boundingRect中的内容
    cv.addWeighted(roi_image, 0, bounding_image, 1, 0, roi_image)
    # 计算旋转中心。此处直接使用放大图像的中点作为旋转中心
    new_center = (enlarged_width // 2, enlarged_height // 2)
    # 向左下/右上倾斜，多转半圈
    if top_most[1]>(x+w)/2:
        angle+=180.0
    # 获取执行旋转所需的变换矩阵
    transform_matrix = cv.getRotationMatrix2D(
        new_center, angle, 1.0)   # 角度为正，表明逆时针旋转
    # 执行旋转
    transformed_image = cv.warpAffine(
        enlarged_image, transform_matrix, (enlarged_width, enlarged_height))

    # 截取与最初等值线框长、宽相同的部分
    output_image = cv.getRectSubPix(
        transformed_image, (rect_width, rect_height), new_center)

    return output_image

# 将车牌图片调整成统一尺寸


def unify_plate_image(plate_image):
    PLATE_STD_HEIGHT = 36			# 车牌区域标准高度
    PLATE_STD_WIDTH = 136			# 车牌区域标准宽度
    uniformed_image = cv.resize(
        plate_image, (PLATE_STD_WIDTH, PLATE_STD_HEIGHT))
    return uniformed_image
