import numpy as np
import cv2 as cv 
import util

HSV_MIN_BLUE_H = 100					# HSV中蓝色分量最小范围值
HSV_MAX_BLUE_H = 140					# HSV中蓝色分量最大范围值
MAX_SV = 250
MIN_SV = 95

plate_file_path = "./img_6.png"
plate_image = cv.imread(plate_file_path)

# 平滑
#plate_image= cv.bilateralFilter(plate_image, 25, 100, 100)
hsv_image = cv.cvtColor(plate_image, cv.COLOR_BGR2HSV)
h_split, s_split, v_split = cv.split(hsv_image)				# 将H,S,V分量分别放到三个数组中
rows, cols = h_split.shape

binary_image = np.zeros((rows, cols), dtype=np.uint8)

# 将满足蓝色背景的区域，对应索引的颜色值置为255，其余置为0，从而实现二值化
for row in np.arange(rows):
    for col in np.arange(cols):
        H = h_split[row, col]
        S = s_split[row, col]
        V = v_split[row, col]
        # 在蓝色值域区间，且满足S和V的一定条件
        if (H >= HSV_MIN_BLUE_H and H<=HSV_MAX_BLUE_H)  \
           and (S >= MIN_SV and S <= MAX_SV)    \
           and (V >= MIN_SV and V <= MAX_SV):          
            binary_image[row, col] = 255
cv.imshow('binary',binary_image)
cv.waitKey()
# 执行闭操作，使相邻区域连成一片
kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 3))
morphology_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)
cv.imshow('morphology',morphology_image)
cv.waitKey()
contours, _ = cv.findContours(morphology_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)



verified_plates = []
for i in np.arange(len(contours)):
    if  util.verify_plate_sizes(contours[i]):
        output_image = util.rotate_plate_image(contours[i], plate_image)
        output_image = util.unify_plate_image(output_image)
        verified_plates.append(output_image)

print(verified_plates)
for i in np.arange(len(verified_plates)):
    cv.imshow("out", verified_plates[i])
    cv.waitKey()

cv.destroyAllWindows()