import numpy as np
import cv2 as cv
from lpr import util

HSV_MIN_BLUE_H = 100  # HSV中蓝色分量最小范围值
HSV_MAX_BLUE_H = 140  # HSV中蓝色分量最大范围值
MAX_SV = 250
MIN_SV = 95


def locate(image):
    # 平滑
    plate_image = cv.bilateralFilter(image, 25, 100, 100)
    hsv_image = cv.cvtColor(plate_image, cv.COLOR_BGR2HSV)
    h_split, s_split, v_split = cv.split(hsv_image)  # 将H,S,V分量分别放到三个数组中
    rows, cols = h_split.shape

    binary_image = np.zeros((rows, cols), dtype=np.uint8)

    # 将满足蓝色背景的区域，对应索引的颜色值置为255，其余置为0，从而实现二值化
    for row in np.arange(rows):
        for col in np.arange(cols):
            H = h_split[row, col]
            S = s_split[row, col]
            V = v_split[row, col]
            # 在蓝色值域区间，且满足S和V的一定条件
            if (H >= HSV_MIN_BLUE_H and H <= HSV_MAX_BLUE_H) \
                    and (S >= MIN_SV and S <= MAX_SV) \
                    and (V >= MIN_SV and V <= MAX_SV):
                binary_image[row, col] = 255

    # 执行闭操作，使相邻区域连成一片
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 3))
    morphology_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(
        morphology_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    verified_plates = []
    for i in np.arange(len(contours)):
        if util.verify_plate_sizes(contours[i]):
            output_image = util.rotate_plate_image(contours[i], plate_image)
            output_image = util.unify_plate_image(output_image)
            verified_plates.append(output_image)

    return verified_plates


if __name__ == "__main__":
    image = cv.imread("./cars/car_example.jpg")

    plates_image = locate(image)

    for i in np.arange(len(plates_image)):
        cv.imwrite('./temp/'+'%s' % i + '.jpg', plates_image[i])
